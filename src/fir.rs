use anyhow::{Context, Result};

use crate::ast::{self, Ast, DIndex, Extra};
use crate::parser::Expr;

/// An index into the `extra` field of `FIR`
type XIndex = usize;

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Op {
    /// function definition
    FunDef {
        name: DIndex,
    },
    FunEnd,
    /// function arg decl
    FunArg,
    /// function call.
    /// fun is a ref to the FIR FunDef node
    /// args points to a slice of Fir Refs in extra
    FunCall {
        fun: Ref,
        // expen
        args: XIndex,
    },
    /// declare a local variable
    /// NOTE: these should be hoisted to the top of the function body
    /// for eaasier codegen
    Alloc,
    Load(Ref),
    Store(Ref, Ref),
    Add(Ref, Ref),
    Sub(Ref, Ref),
    Mul(Ref, Ref),
    Div(Ref, Ref),
    Eq(Ref, Ref),
    Lt(Ref, Ref),
    LtEq(Ref, Ref),
    Gt(Ref, Ref),
    GtEq(Ref, Ref),
    Branch {
        cond: Ref,
        t: Ref,
        f: Ref,
    },
    Jump(Ref),
}

pub struct FIR {
    ops: Vec<Op>,
    types: Vec<TypeRef>,
    extra: Extra,
    data: ast::DataPool,
}

impl FIR {
    fn new(ops: Vec<Op>, extra: Extra, data: ast::DataPool, types: Vec<TypeRef>) -> Self {
        return Self { ops, extra, data, types };
    }

    fn get_const(&self, i: u32) -> u64 {
        let di = self.extra[i as usize];
        return self.data.get::<u64>(di as usize);
    }
}

pub struct FIRGen {
    ast: Ast,
    cursor: usize,

    ops: Vec<Op>,
    types: Vec<TypeRef>,
    extra: Extra,

    scopes: ScopeStack,
}

impl FIRGen {
    pub fn generate(ast: Ast) -> Result<FIR> {
        let mut this = FIRGen {
            ast,
            cursor: 0,
            ops: vec![],
            types: vec![],
            extra: Extra::new(),
            scopes: ScopeStack::new(),
        };
        this._generate()?;

        return Ok(FIR::new(this.ops, this.extra, this.ast.data, this.types));
    }

    fn _generate(&mut self) -> Result<()> {
        while self.cursor < self.ast.exprs.len() {
            self.gen_expr(self.cursor)?;
            self.cursor += 1;
        }
        Ok(())
    }

    fn push(&mut self, op: Op) -> usize {
        let i = self.ops.len();
        self.ops.push(op);
        self.types.push(TypeRef::None);
        self.scopes.skip::<1>();
        debug_assert_eq!(self.ops.len(), self.types.len());
        return i;
    }

    fn push_typed(&mut self, op: Op, ty: TypeRef) -> usize {
        let i = self.ops.len();
        self.ops.push(op);
        self.types.push(ty);
        self.scopes.skip::<1>();
        debug_assert_eq!(self.ops.len(), self.types.len());
        return i;
    }

    fn push_named(&mut self, op: Op, name: DIndex, ty: TypeRef) -> usize {
        let i = self.ops.len();
        self.ops.push(op);
        self.types.push(ty);
        self.scopes.bind_local(name);
        debug_assert_eq!(self.ops.len(), self.types.len());
        return i;
    }

    fn mark_visited(&mut self, cursor: usize) {
        if cursor <= self.cursor {
            return;
        }
        self.cursor = cursor;
    }

    fn gen_expr(&mut self, i: usize) -> Result<usize> {
        let expr = self.ast.exprs[i];
        self.mark_visited(i);
        match expr {
            Expr::Binop { op, lhs, rhs } => self.gen_binop(op, lhs, rhs),
            Expr::Ident(name) => {
                let i = self.resolve_ident(name)?;
                let load_i = self.push(Op::Load(i));
                return Ok(load_i);
            }
            Expr::Int(val) => {
                // TODO: remove this inderection
                let i = self.extra.append_u32(val as u32);
                let ty = TypeRef::IntU64;
                let const_op = Ref::Const(ty, i);
                let load_op = Op::Load(const_op);
                let op_i = self.push(load_op);
                return Ok(op_i);
            }
            Expr::Bind { name, value } => {
                let ty = TypeRef::IntU64;
                // FIXME: hoist!
                let alloc_i = self.push_named(Op::Alloc, name, ty);
                let value_i = self.gen_expr(value)?;
                let alloc_ref = Ref::Inst(alloc_i as u32);
                let value_ref = Ref::Inst(value_i as u32);
                let store_op = Op::Store(alloc_ref, value_ref);
                let store_i = self.push(store_op);
                return Ok(store_i);
            }
            Expr::FunDef { name, args, body } => {
                let fun_def_op = Op::FunDef { name };
                let ty = TypeRef::IntU64;
                let fun_def_i = self.push_typed(fun_def_op, ty);
                let n_args = self.ast.get_num_args(args);
                for _ in 0..n_args {
                    let ty = TypeRef::IntU64;
                    self.push_typed(Op::FunArg, ty);
                }
                self.gen_expr(body)?;
                self.push(Op::FunEnd);
                return Ok(fun_def_i);
            }
            _ => unimplemented!("unimplemented expr {:?}", expr),
        }
    }

    fn gen_binop(&mut self, op: crate::parser::Binop, lhs: usize, rhs: usize) -> Result<usize> {
        let lhs = Ref::Inst(self.gen_expr(lhs)? as u32);
        let rhs = Ref::Inst(self.gen_expr(rhs)? as u32);

        use crate::parser::Binop::*;

        let op_inst = match op {
            Add => Op::Add(lhs, rhs),
            Sub => Op::Sub(lhs, rhs),
            Mul => Op::Mul(lhs, rhs),
            _ => unimplemented!("unimplemented binop {:?}", op),
        };
        let i = self.push(op_inst);
        Ok(i)
    }

    fn resolve_ident(&self, i: DIndex) -> Result<Ref> {
        let i = self.scopes.get(i).context("undefined variable")?;
        return Ok(Ref::Inst(i));
    }
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub enum Ref {
    Inst(u32),
    Const(TypeRef, u32),
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub enum TypeRef {
    None,
    IntU64,
}

struct ScopeStack {
    stack_map: Vec<DIndex>,
    starts: Vec<usize>,
    cur: usize,
}

impl ScopeStack {
    const NOT_BOUND: DIndex = usize::MAX;

    fn new() -> Self {
        Self {
            stack_map: vec![],
            starts: vec![0],
            cur: 0,
        }
    }

    fn skip<const N: usize>(&mut self) {
        let vals = [Self::NOT_BOUND; N];
        self.stack_map.extend(vals);
    }

    /// like start_new but does not set `cur`
    fn start_subscope(&mut self) {
        self.starts.push(self.stack_map.len());
    }

    fn end_subscope(&mut self) {
        let start = self.starts.pop().expect("no starts");
        self.stack_map.truncate(start);
    }

    fn start_new(&mut self) {
        self.starts.push(self.cur);
        self.cur = self.stack_map.len();
    }

    fn end(&mut self) {
        let start = self.cur;
        self.cur = self.starts.pop().expect("no starts");
        self.stack_map.truncate(start);
    }

    /// NOTE: does not check that name is not already in scope,
    /// therefore allowing shadowing
    fn bind_local(&mut self, name: DIndex) -> u32 {
        let i = self.stack_map.len();
        self.stack_map.push(name);
        return (i - self.cur) as u32;
    }

    fn get(&self, name: DIndex) -> Option<u32> {
        debug_assert_ne!(name, Self::NOT_BOUND);
        let pos_r = self.stack_map.iter().rev().position(|&n| n == name);
        let Some(pos_r) = pos_r else {
            return None;
        };
        let pos = self.stack_map.len() - pos_r - 1;
        if pos < self.cur {
            unimplemented!("tried to get reference to variable outside of stack frame. globals not implemented");
        }
        return Some(pos as u32);
    }

    fn set(&mut self, i: u32, name: DIndex) {
        self.stack_map[i as usize] = name;
    }

    fn is_last(&self, i: u32) -> bool {
        let i = self.cur + i as usize;
        let len = self.stack_map.len();
        return i + 1 == len;
    }
}
struct FIRStringifier<'fir> {
    fir: &'fir FIR,
    str: String,
    in_func: bool,
}

#[allow(dead_code)]
impl<'fir> FIRStringifier<'fir> {
    const INDENT: &'static str = "  ";

    fn stringify(fir: &'fir FIR) -> String {
        let this = Self {
            fir,
            str: String::new(),
            in_func: false,
        };
        return this._stringify();
    }

    fn _stringify(mut self) -> String {
        for (i, inst) in self.fir.ops.iter().enumerate() {
            match inst {
                Op::Load(r) => {
                    self.inst_eq(i);
                    self.func_1("load", |s| s.write_ref(r));
                }
                Op::Add(lhs, rhs) | Op::Sub(lhs, rhs) | Op::Mul(lhs, rhs) => {
                    let name = self.get_op_name(*inst);
                    self.inst_eq(i);
                    self.func_2(name, |s| s.write_ref(lhs), |s| s.write_ref(rhs));
                }
                Op::FunDef { name } => {
                    assert!(!self.in_func, "nested functions not supported");
                    self.write("define");
                    self.space();
                    let return_ty = &self.fir.types[i];
                    self.write_type_ref(return_ty);
                    self.space();
                    self.func_ref(*name);
                    self.space();
                    self.write("{");
                    self.in_func = true;
                }
                Op::FunArg => {
                    assert!(self.in_func, "arg decl outside of function");
                    self.inst_eq(i);
                    let ty = &self.fir.types[i];
                    self.func_1("arg", |s| s.write_type_ref(ty));
                }
                Op::FunEnd => {
                    self.in_func = false;
                    self.write("}");
                }
                Op::Alloc => {
                    self.inst_eq(i);
                    let ty = &self.fir.types[i];
                    self.func_1("alloc", |s| s.write_type_ref(ty));
                }
                Op::Store(dest, src) => {
                    self.func_2("store", |s| s.write_ref(dest), |s| s.write_ref(src));
                }
                _ => unimplemented!("FIR op {:?} not implemented", inst),
            };
            if i < self.fir.ops.len() {
                self.newline();
            }
        }
        return self.str;
    }

    fn write(&mut self, str: &str) {
        self.str.push_str(str);
    }

    fn newline(&mut self) {
        self.write("\n");
        if self.in_func {
            self.write(Self::INDENT);
        }
    }

    fn space(&mut self) {
        self.write(" ");
    }

    fn func_1<F>(&mut self, name: &str, arg: F)
    where
        F: FnOnce(&mut Self),
    {
        self.write(name);
        self.write("(");
        arg(self);
        self.write(")");
    }

    fn func_2<F1, F2>(&mut self, name: &str, arg1: F1, arg2: F2)
    where
        F1: FnOnce(&mut Self),
        F2: FnOnce(&mut Self),
    {
        self.write(name);
        self.write("(");
        arg1(self);
        self.write(", ");
        arg2(self);
        self.write(")");
    }

    fn inst_ref(&mut self, i: u32) {
        self.write("%");
        self.write(i.to_string().as_str());
    }

    fn inst_eq(&mut self, i: usize) {
        self.inst_ref(i as u32);
        self.write(" = ");
    }

    fn func_ref(&mut self, i: DIndex) {
        self.write("@");
        self.write_ident(i);
    }

    fn write_ref(&mut self, r: &Ref) {
        match r {
            Ref::Const(ty, i) => self.func_2(
                "const",
                |s| s.write_type_ref(ty),
                |s| s.write(&self.fir.get_const(*i).to_string()),
            ),
            Ref::Inst(i) => self.inst_ref(*i),
        }
    }

    fn write_type_ref(&mut self, ty: &TypeRef) {
        use TypeRef::*;
        let str = match ty {
            IntU64 => "u64",
            None => "_"
        };
        self.write(str);
    }

    fn write_ident(&mut self, i: DIndex) {
        let str = self.fir.data.get_ref::<str>(i);
        self.write(str);
    }

    fn get_op_name(&self, op: Op) -> &'static str {
        use Op::*;
        match op {
            FunDef { .. } => "define",
            FunEnd => "fun_end",
            FunArg => "arg",
            FunCall { .. } => "call",
            Alloc => "alloc",
            Load(_) => "load",
            Store(_, _) => "store",
            Add(_, _) => "add",
            Sub(_, _) => "sub",
            Mul(_, _) => "mul",
            Div(_, _) => "div",
            Eq(_, _) => "eq",
            Lt(_, _) => "lt",
            LtEq(_, _) => "lteq",
            Gt(_, _) => "gt",
            GtEq(_, _) => "gteq",
            Branch { .. } => "br",
            Jump(_) => "jmp",
        }
    }

    fn op_name(&mut self, op: Op) {
        let str = self.get_op_name(op);
        self.write(str);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_matches {
        ($expr:expr, $pat:pat) => {
            assert!(
                matches!($expr, $pat),
                "expected {:?}, got {:?}",
                stringify!($pat),
                $expr
            )
        };
        ($expr:expr, $pat:pat, $message:literal) => {
            assert!(matches!($expr, $pat), $message)
        };
        ($expr:expr, $pat:pat => $body:expr) => {{
            assert_matches!($expr, $pat);
            match $expr {
                $pat => $body,
                _ => unreachable!("unreachable pattern"),
            }
        }};
    }

    macro_rules! assert_fir_matches {
        ($fir:expr, [$($ops:pat),*]) => {
            let mut i = 0;
            $(
                #[allow(unused_assignments)]
                {
                    assert_matches!($fir.ops[i], $ops);
                    i += 1;
                }
            )*
            assert_eq!($fir.ops.len(), i, "expected {} ops, got {}. Extra: {:?}", i, $fir.ops.len(), &$fir.ops[i..]);
        };
    }

    macro_rules! assert_results_in_fir {
        ($contents:expr, [$($ops:pat),*]) => {
            let contents = $contents;
            let parser = crate::parser::Parser::new(contents);
            let ast = parser.parse().expect("syntax error");
            let fir = FIRGen::generate(ast).expect("failed to generate fir");
            assert_fir_matches!(fir, [$($ops),*]);
        };
    }

    #[test]
    fn add() {
        let contents = "( + 1 2 )";

        use Op::*;

        assert_results_in_fir!(
            contents,
            [
                Load(Ref::Const(TypeRef::IntU64, 0)),
                Load(Ref::Const(TypeRef::IntU64, 1)),
                Add(Ref::Inst(0), Ref::Inst(1))
            ]
        );
    }

    macro_rules! assert_fir_str_eq {
        ($contents:literal, $($lines:literal),*) => {
            let parser = crate::parser::Parser::new($contents);
            let ast = parser.parse().expect("syntax error");
            let fir = FIRGen::generate(ast).expect("failed to generate fir");
            let fir_str = FIRStringifier::stringify(&fir);
            let mut fir_lines = fir_str.lines();
            #[allow(unused)]
            let mut i = 0;
            $(
                #[allow(unused_assignments)]
                {
                    assert_eq!(fir_lines.next().unwrap(), $lines);
                    i += 1
                }
            )*
        };
    }

    mod stringify {
        use super::*;

        #[test]
        fn add() {
            assert_fir_str_eq!(
                "( + 1 2)",
                "%0 = load(const(u64, 1))",
                "%1 = load(const(u64, 2))",
                "%2 = add(%0, %1)"
            );
        }

        #[test]
        fn chained_algebra() {
            assert_fir_str_eq!(
                "(+ 0 ( + 1 ( * 2 (- 3 4))))",
                "%0 = load(const(u64, 0))",
                "%1 = load(const(u64, 1))",
                "%2 = load(const(u64, 2))",
                "%3 = load(const(u64, 3))",
                "%4 = load(const(u64, 4))",
                "%5 = sub(%3, %4)",
                "%6 = mul(%2, %5)",
                "%7 = add(%1, %6)",
                "%8 = add(%0, %7)"
            );
        }

        #[test]
        fn if_expr() {
            assert_fir_str_eq!(
                "(if (== 1 2) 3 4)",
                "%0 = load(const(u64, 1))",
                "%1 = load(const(u64, 2))",
                "%2 = cmp_eq(u64, %0, %1)",
                "br(%2, %3, %4)",
                "%3 = label()",
                "jmp(%5)",
                "%4 = label()",
                "jmp(%5)",
                "%5 = label()",
                "%6 = load(phi(u64, [%3, const(u64, 3)], [%4, const(u64, 4)]))"
            );
        }

        #[test]
        fn fundef() {
            assert_fir_str_eq!(
                "(fun add (a b) (+ a b))",
                "define u64 @add {",
                "  %0 = arg(u64)",
                "  %1 = arg(u64)",
                "  %2 = add(%0, %1)",
                "  %3 = ret(%2)",
                "}"
            );
        }

        #[test]
        fn funcall() {
            assert_fir_str_eq!(
                "(fun add (a b) (+ a b)) (add 1 2)",
                "define u64 @add {",
                "  %0 = arg(u64)",
                "  %1 = arg(u64)",
                "  %2 = add(%0, %1)",
                "  %3 = ret(%2)",
                "}",
                "%0 = load(const(u64, 1))",
                "%1 = load(const(u64, 2))",
                "%2 = call(@add, [%0, %1])"
            );
        }

        #[test]
        fn bind() {
            assert_fir_str_eq!(
                "(let a 1)",
                "%0 = alloc(u64)",
                "%1 = load(const(u64, 1))",
                "store(%0, %1)"
            );
        }

        #[test]
        fn bind_use() {
            assert_fir_str_eq!(
                "(let a 1) a",
                "%0 = alloc(u64)",
                "%1 = load(const(u64, 1))",
                "store(%0, %1)",
                "%3 = load(%0)"
            );
        }
    }
}
