use anyhow::{anyhow, Context, Result};

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
    Ret(Ref),
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
    Phi {
        a: (Ref, Ref),
        b: (Ref, Ref)
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
        return Self {
            ops,
            extra,
            data,
            types,
        };
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
        return i - self.scopes.cur;
    }

    fn push_typed(&mut self, op: Op, ty: TypeRef) -> usize {
        let i = self.ops.len();
        self.ops.push(op);
        self.types.push(ty);
        self.scopes.skip::<1>();
        debug_assert_eq!(self.ops.len(), self.types.len());
        return i - self.scopes.cur;
    }

    fn push_named(&mut self, op: Op, name: DIndex, ty: TypeRef) -> usize {
        let i = self.ops.len();
        self.ops.push(op);
        self.types.push(ty);
        self.scopes.bind_local(name);
        debug_assert_eq!(self.ops.len(), self.types.len());
        return i - self.scopes.cur;
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
            Expr::Binop { op, lhs, rhs } => {
                let lhs_i = self.gen_expr(lhs)?;
                let lhs = Ref::Inst(lhs_i as u32);

                let rhs_i = self.gen_expr(rhs)?;
                let rhs = Ref::Inst(rhs_i as u32);

                use crate::parser::Binop::*;
                let op_inst = match op {
                    Add => Op::Add(lhs, rhs),
                    Sub => Op::Sub(lhs, rhs),
                    Mul => Op::Mul(lhs, rhs),
                    Eq => Op::Eq(lhs, rhs),
                    _ => unimplemented!("unimplemented binop {:?}", op),
                };
                let op_ty = self.join_types(lhs_i, rhs_i);

                let i = self.push_typed(op_inst, op_ty);
                Ok(i)
            },
            Expr::Ident(name) => {
                let i = self.resolve_ident(name)?;
                let ty = self.types[i as usize];
                let load_op = Op::Load(Ref::Inst(i));
                // propogates the type
                let load_i = self.push_typed(load_op, ty);
                return Ok(load_i);
            }
            Expr::Int(val) => {
                // TODO: remove this inderection
                let i = self.extra.append_u32(val as u32);
                let const_op = Ref::Const(i);
                let load_op = Op::Load(const_op);
                let ty = TypeRef::IntU64;
                let op_i = self.push_typed(load_op, ty);
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
                self.scopes.start_new();
                // FIXME: `to_owned` here because rust cannot determine that
                // `self.push_named` does not mutate the contents of `args_iter`
                let args_iter = self.ast.fun_args_slice(args).to_owned().into_iter();
                for arg in args_iter {
                    let ty = TypeRef::IntU64;
                    let arg = arg as usize;
                    self.push_named(Op::FunArg, arg, ty);
                }
                let res = self.gen_expr(body)?;
                self.push(Op::Ret(Ref::Inst(res as u32)));
                self.push(Op::FunEnd);
                self.scopes.end();
                return Ok(fun_def_i);
            }
            Expr::FunCall { name, args } => {
                // NOTE: to_owned here because rust cannot determine that
                // gen_expr does not mutate the contents of args
                let mut args = self.ast.fun_args_slice(args).to_owned();
                let mut args_iter = args.iter_mut();
                for arg_expr_i in &mut args_iter {
                    // overwrite index in args
                    // cannot append to extra here because arg exprs may 
                    // add data to extra
                    *arg_expr_i = self.gen_expr(*arg_expr_i as usize)? as u32;
                }
                // NOTE: concat stores len, but len could be fetched on demand
                // from fundef instead of stored again
                let args_i = self.extra.concat(&args);

                let fun_i = self.resolve_fn(name)?;
                let fun_ref = Ref::Inst(fun_i);

                let fun_op = Op::FunCall {
                    fun: fun_ref,
                    args: args_i,
                };
                let ret_ty = self.types[fun_i as usize];
                let res = self.push_typed(fun_op, ret_ty);
                return Ok(res);
            }
            Expr::If { cond, branch_true, branch_false } => {
                let cond_i = self.gen_expr(cond)?;
                let cond_ref = Ref::Inst(cond_i as u32);
                let br_i = self.reserve();

                let true_i = self.next_i();
                let true_res_i = self.gen_expr(branch_true)?;
                let true_end_i = self.reserve();
                let true_ref = Ref::Inst(true_i as u32);
                let true_res_ref = Ref::Inst(true_res_i as u32);

                let false_i = self.next_i();
                let false_res_i = self.gen_expr(branch_false)?;
                let false_end_i = self.reserve();
                let false_ref = Ref::Inst(false_i as u32);
                let false_res_ref = Ref::Inst(false_res_i as u32);

                let branch_op = Op::Branch { cond: cond_ref, t: true_ref, f: false_ref};
                self.set(br_i, branch_op);

                let res_op = Op::Phi {
                    a: (true_ref, true_res_ref),
                    b: (false_ref, false_res_ref),
                };
                let res_ty = self.join_types(true_res_i, false_res_i);
                let res_i = self.push_typed(res_op, res_ty);

                let jmp_res_op = Op::Jump(Ref::Inst(res_i as u32));
                self.set(true_end_i, jmp_res_op);
                self.set(false_end_i, jmp_res_op);

                return Ok(res_i);
            }
            _ => unimplemented!("unimplemented expr {:?}", expr),
        }
    }

    fn resolve_ident(&self, name: DIndex) -> Result<u32> {
        let i = self.scopes.get(name).context("undefined variable")?;
        return Ok(i);
    }

    fn resolve_fn(&self, name: DIndex) -> Result<u32> {
        // FIXME: create map
        for (i, op) in self.ops.iter().enumerate() {
            let Op::FunDef { name: n } = op else {
                continue;
            };
            if *n == name {
                return Ok(i as u32);
            }
        }
        return Err(anyhow!("undefined function"));
    }

    fn reserve(&mut self) -> usize {
        return self.push(Op::Alloc);
    }

    fn set(&mut self, i: usize, op: Op) {
        self.ops[i] = op;
    }

    fn next_i(&self) -> usize {
        return self.ops.len();
    }

    fn join_types(&self, a: usize, b: usize) -> TypeRef {
        let a = self.types[a];
        let b = self.types[b];
        if a == b {
            return a;
        }
        let none = TypeRef::None;
        if a == none {
            return b;
        }
        if b == none {
            return a;
        }
        unimplemented!("unimplemented type mismatch");
    }
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub enum Ref {
    Inst(u32),
    Const(u32),
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TypeRef {
    None,
    IntU64,
}

#[derive(Debug)]
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
        return Some((pos - self.cur) as u32);
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
    cur_func_i: Option<usize>,
    offset: u32,
}

#[allow(dead_code)]
impl<'fir> FIRStringifier<'fir> {
    const INDENT: &'static str = "  ";

    fn stringify(fir: &'fir FIR) -> String {
        let this = Self {
            fir,
            str: String::new(),
            cur_func_i: None,
            offset: 0,
        };
        return this._stringify();
    }

    fn _stringify(mut self) -> String {
        for (i, inst) in self.fir.ops.iter().enumerate() {
            match inst {
                Op::Load(r) => {
                    self.inst_eq(i);
                    let ty = &self.fir.types[i];
                    self.op_func_2_ty_ref(inst, ty, r);
                }
                Op::Add(lhs, rhs) | Op::Sub(lhs, rhs) | Op::Mul(lhs, rhs) => {
                    self.inst_eq(i);
                    self.op_func_2_ref_ref(inst, lhs, rhs);
                }
                Op::FunDef { name } => {
                    assert!(!self.in_func(), "nested functions not supported");
                    self.write("define");
                    self.space();
                    let return_ty = &self.fir.types[i];
                    self.write_type_ref(return_ty);
                    self.space();
                    self.func_ref(*name);
                    self.space();
                    self.write("{");
                    self.cur_func_i = Some(i + 1);
                }
                Op::FunArg => {
                    assert!(self.in_func(), "arg decl outside of function");
                    self.inst_eq(i);
                    let ty = &self.fir.types[i];
                    self.op_func_1_ty(inst, ty);
                }
                Op::FunCall { fun, args } => {
                    let Ref::Inst(fun_i) = fun else {
                        unreachable!("fun call ref not inst");
                    };
                    let Op::FunDef { name } = &self.fir.ops[*fun_i as usize] else {
                        unreachable!("fun call ref not fun def");
                    };
                    self.inst_eq(i);
                    self.op_name(*inst); // "call"
                    self.paren_start();
                    let ret_ty = &self.fir.types[i];
                    self.write_type_ref(ret_ty);
                    self.sep();
                    self.func_ref(*name);
                    self.sep();
                    self.brack_start();
                    let ast::ExtraFunArgs { args } = self.fir.extra.get::<ast::ExtraFunArgs>(*args);
                    for arg in args.iter().take(args.len() - 1) {
                        self.inst_ref(*arg);
                        self.sep();
                    }
                    self.inst_ref(*args.last().unwrap());
                    self.brack_end();
                    self.paren_end();
                }
                Op::Ret(r) => {
                    self.inst_eq(i);
                    self.op_func_1_ref(inst, r);
                }
                Op::FunEnd => {
                    self.write("}");
                }
                Op::Alloc => {
                    self.inst_eq(i);
                    let ty = &self.fir.types[i];
                    self.op_func_1_ty(inst, ty);
                }
                Op::Store(dest, src) => {
                    self.op_func_2_ref_ref(inst, dest, src);
                }
                Op::Eq(lhs, rhs) => {
                    self.inst_eq(i);
                    let ty = &self.fir.types[i];
                    self.op_func_3_ty_ref_ref(inst,ty, lhs, rhs);
                }
                Op::Branch { cond, t, f } => {
                    self.op_func_3_ref_ref_ref(inst, cond, t, f);
                }
                Op::Jump(dest) => {
                    self.op_func_1_ref(inst, dest);
                }
                Op::Phi { a: (a_from, a_res), b: (b_from, b_res)} => {
                    self.inst_eq(i);
                    self.op_name(*inst);
                    self.paren_start();

                    self.write_type_ref_at(i);

                    self.sep();

                    self.brack_start();
                    self.write_ref(a_from);
                    self.sep();
                    self.write_ref(a_res);
                    self.brack_end();

                    self.sep();

                    self.brack_start();
                    self.write_ref(b_from);
                    self.sep();
                    self.write_ref(b_res);
                    self.brack_end();

                    self.paren_end();

                }
                _ => unimplemented!("FIR op {:?} not implemented", inst),
            };
            let next_is_fun_end = matches!(self.fir.ops.get(i + 1), Some(Op::FunEnd));
            if next_is_fun_end {
                // set here so the newline is not printed with indent before closing brace
                let func_start = self.cur_func_i.take().expect("in function before func end");
                // NOTE: +3 for the +1 offset of func_start, funEnd, and ???
                self.offset += (i - func_start + 3) as u32;
            }
            let not_at_last_op = i < self.fir.ops.len();
            if not_at_last_op {
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
        if self.in_func() {
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
        self.paren_start();
        arg(self);
        self.paren_end();
    }

    fn op_func_1<F>(&mut self, op: Op, arg: F)
    where
        F: FnOnce(&mut Self),
    {
        let name = self.get_op_name(op);
        self.func_1(name, arg);
    }

    fn op_func_1_ref(&mut self, op: &Op, arg: &Ref) {
        let name = self.get_op_name(*op);
        self.write(name);
        self.paren_start();
        self.write_ref(arg);
        self.paren_end();
    }

    fn op_func_1_ty(&mut self, op: &Op, arg: &TypeRef) {
        let name = self.get_op_name(*op);
        self.write(name);
        self.paren_start();
        self.write_type_ref(arg);
        self.paren_end();
    }

    fn func_2<F1, F2>(&mut self, name: &str, arg1: F1, arg2: F2)
    where
        F1: FnOnce(&mut Self),
        F2: FnOnce(&mut Self),
    {
        self.write(name);
        self.paren_start();
        arg1(self);
        self.sep();
        arg2(self);
        self.paren_end();
    }

    fn op_func_2<F1, F2>(&mut self, op: Op, arg1: F1, arg2: F2)
    where
        F1: FnOnce(&mut Self),
        F2: FnOnce(&mut Self),
    {
        let name = self.get_op_name(op);
        self.func_2(name, arg1, arg2);
    }
    fn op_func_2_ref_ref(&mut self, op: &Op, arg1: &Ref, arg2: &Ref) {
        let name = self.get_op_name(*op);
        self.write(name);
        self.paren_start();
        self.write_ref(arg1);
        self.sep();
        self.write_ref(arg2);
        self.paren_end();
    }

    fn op_func_2_ty_ref(&mut self, op: &Op, arg1: &TypeRef, arg2: &Ref) {
        let name = self.get_op_name(*op);
        self.write(name);
        self.paren_start();
        self.write_type_ref(arg1);
        self.sep();
        self.write_ref(arg2);
        self.paren_end();
    }

    fn op_func_3_ty_ref_ref(&mut self, op: &Op, ty: &TypeRef, a: &Ref, b: &Ref) {
        self.op_name(*op);
        self.paren_start();
        self.write_type_ref(ty);
        self.sep();
        self.write_ref(a);
        self.sep();
        self.write_ref(b);
        self.paren_end();
    }

    fn op_func_3_ref_ref_ref(&mut self, op: &Op, a: &Ref, b: &Ref, c: &Ref) {
        self.op_name(*op);
        self.paren_start();
        self.write_ref(a);
        self.sep();
        self.write_ref(b);
        self.sep();
        self.write_ref(c);
        self.paren_end();
    }

    fn inst_ref(&mut self, i: u32) {
        self.write("%");
        let i = i - self.offset;
        self.write(i.to_string().as_str());
    }

    fn inst_eq(&mut self, mut i: usize) {
        if let Some(func_offset) = self.cur_func_i {
            i -= func_offset;
        }
        self.inst_ref(i as u32);
        self.write(" = ");
    }

    fn func_ref(&mut self, i: DIndex) {
        self.write("@");
        self.write_ident(i);
    }

    fn write_ref(&mut self, r: &Ref) {
        match r {
            Ref::Const(i) => self.func_1("const", |s| s.write(&self.fir.get_const(*i).to_string())),
            Ref::Inst(i) => self.inst_ref(*i),
        }
    }

    fn write_type_ref(&mut self, ty: &TypeRef) {
        use TypeRef::*;
        let str = match ty {
            IntU64 => "u64",
            None => "_",
        };
        self.write(str);
    }

    fn write_type_ref_at(&mut self, i: usize) {
        let ty = &self.fir.types[i];
        self.write_type_ref(ty);
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
            Ret(_) => "ret",
            Alloc => "alloc",
            Load(_) => "load",
            Store(_, _) => "store",
            Add(_, _) => "add",
            Sub(_, _) => "sub",
            Mul(_, _) => "mul",
            Div(_, _) => "div",
            Eq(_, _) => "cmp_eq",
            Lt(_, _) => "cmp_lt",
            LtEq(_, _) => "cmp_lteq",
            Gt(_, _) => "cmp_gt",
            GtEq(_, _) => "cmp_gteq",
            Branch { .. } => "br",
            Jump(_) => "jmp",
            Phi {..} => "phi",
        }
    }

    fn op_name(&mut self, op: Op) {
        let str = self.get_op_name(op);
        self.write(str);
    }

    fn in_func(&self) -> bool {
        return self.cur_func_i.is_some();
    }

    fn brack_start(&mut self) {
        self.write("[");
    }

    fn brack_end(&mut self) {
        self.write("]");
    }

    fn paren_start(&mut self) {
        self.write("(");
    }

    fn paren_end(&mut self) {
        self.write(")");
    }

    fn sep(&mut self) {
        self.write(", ");
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

    fn gen_fir(contents: &str) -> FIR {
        let parser = crate::parser::Parser::new(contents);
        let ast = parser.parse().expect("syntax error");
        let fir = FIRGen::generate(ast).expect("failed to generate fir");
        return fir;
    }

    macro_rules! assert_results_in_fir {
        ($contents:expr, [$($ops:pat),*]) => {
            let fir = gen_fir($contents);
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
                Load(Ref::Const(0)),
                Load(Ref::Const(1)),
                Add(Ref::Inst(0), Ref::Inst(1))
            ]
        );
    }

    #[test]
    fn fundef() {
        use Op::*;
        assert_results_in_fir!(
            "(fun add (a b) (+ a b))",
            [
                FunDef { .. },
                FunArg,
                FunArg,
                Load(Ref::Inst(0)),
                Load(Ref::Inst(1)),
                Add(Ref::Inst(2), Ref::Inst(3)),
                Ret(Ref::Inst(4)),
                FunEnd
            ]
        );
    }
    #[test]
    fn funcall() {
        use Op::*;
        let contents = "(fun add (a b) (+ a b)) (add 1 2)";
        assert_results_in_fir!(
            contents,
            [
                FunDef { .. },
                FunArg,
                FunArg,
                Load(Ref::Inst(0)),
                Load(Ref::Inst(1)),
                Add(Ref::Inst(2), Ref::Inst(3)),
                Ret(Ref::Inst(4)),
                FunEnd,
                Load(Ref::Const(0)),
                Load(Ref::Const(1)),
                FunCall {
                    fun: Ref::Inst(0),
                    args: 2
                }
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
                    assert_eq!($lines, fir_lines.next().unwrap());
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
                "%0 = load(u64, const(1))",
                "%1 = load(u64, const(2))",
                "%2 = add(%0, %1)"
            );
        }

        #[test]
        fn chained_algebra() {
            assert_fir_str_eq!(
                "(+ 0 ( + 1 ( * 2 (- 3 4))))",
                "%0 = load(u64, const(0))",
                "%1 = load(u64, const(1))",
                "%2 = load(u64, const(2))",
                "%3 = load(u64, const(3))",
                "%4 = load(u64, const(4))",
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
                "%0 = load(u64, const(1))",
                "%1 = load(u64, const(2))",
                "%2 = cmp_eq(u64, %0, %1)",
                "br(%2, %4, %6)",
                "%4 = load(u64, const(3))",
                "jmp(%8)",
                "%6 = load(u64, const(4))",
                "jmp(%8)",
                "%8 = phi(u64, [%4, %4], [%6, %6])"
            );
        }

        #[test]
        fn fundef() {
            assert_fir_str_eq!(
                "(fun add (a b) (+ a b))",
                "define u64 @add {",
                "  %0 = arg(u64)",
                "  %1 = arg(u64)",
                "  %2 = load(u64, %0)",
                "  %3 = load(u64, %1)",
                "  %4 = add(%2, %3)",
                "  %5 = ret(%4)",
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
                "  %2 = load(u64, %0)",
                "  %3 = load(u64, %1)",
                "  %4 = add(%2, %3)",
                "  %5 = ret(%4)",
                "}",
                "%0 = load(u64, const(1))",
                "%1 = load(u64, const(2))",
                "%2 = call(u64, @add, [%0, %1])"
            );
        }

        #[test]
        fn bind() {
            assert_fir_str_eq!(
                "(let a 1)",
                "%0 = alloc(u64)",
                "%1 = load(u64, const(1))",
                "store(%0, %1)"
            );
        }

        #[test]
        fn bind_use() {
            assert_fir_str_eq!(
                "(let a 1) a",
                "%0 = alloc(u64)",
                "%1 = load(u64, const(1))",
                "store(%0, %1)",
                "%3 = load(u64, %0)"
            );
        }
    }
}
