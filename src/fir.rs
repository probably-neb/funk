use anyhow::{Context, Result};

use crate::ast::{self, Ast, DIndex, Extra};
use crate::parser::Expr;

/// An index into the `extra` field of `FIR`
type XIndex = usize;

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Op {
    /// function definition
    FunDef,
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
    Store(Ref),
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

impl Op {
    fn name(&self) -> &'static str {
        use Op::*;
        match self {
            FunDef => "def",
            FunArg => "arg",
            FunCall { .. } => "call",
            Alloc => "alloc",
            Load(_) => "load",
            Store(_) => "store",
            Add(_, _) => "add",
            Sub(_, _) => "sub",
            Mul(_, _) => "mul",
            Div(_, _) => "div",
            Eq(_, _) => "eq",
            Lt(_, _) => "lt",
            LtEq(_, _) => "lteq",
            Gt(_, _) => "gt",
            GtEq(_, _) => "gteq",
            Branch { .. } => "branch",
            Jump(_) => "jump",
        }
    }
}

pub struct FIR {
    ops: Vec<Op>,
    extra: Extra,
    data: ast::DataPool,
}

impl FIR {
    fn new(ops: Vec<Op>, extra: Extra, data: ast::DataPool) -> Self {
        return Self { ops, extra, data };
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
    extra: Extra,
}

impl FIRGen {
    pub fn generate(ast: Ast) -> Result<FIR> {
        let mut this = FIRGen {
            ast,
            cursor: 0,
            ops: vec![],
            extra: Extra::new(),
        };
        this._generate()?;

        return Ok(FIR::new(this.ops, this.extra, this.ast.data));
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

            Expr::Int(val) => {
                let i = self.extra.append_u32(val as u32);
                let op_i = self.push(Op::Load(Ref::Const(TypeRef::IntU64, i)));
                return Ok(op_i);
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
}

#[derive(Debug, Copy, Clone)]
pub enum Ref {
    Inst(u32),
    Const(TypeRef, u32),
}

#[derive(Debug, Copy, Clone)]
pub enum TypeRef {
    IntU64,
}

struct FIRStringifier<'fir> {
    fir: &'fir FIR,
    str: String,
}

impl<'fir> FIRStringifier<'fir> {
    fn stringify(fir: &'fir FIR) -> String {
        let mut this = Self {
            fir,
            str: String::new(),
        };
        return this._stringify();
    }

    fn _stringify(mut self) -> String {
        for (i, inst) in self.fir.ops.iter().enumerate() {
            match inst {
                Op::Load(r) => {
                    self.inst_i_eq(i);
                    self.func_1("load", |s| s.write_ref(r));
                }
                Op::Add(lhs, rhs) | Op::Sub(lhs, rhs) | Op::Mul(lhs, rhs) => {
                    let name = inst.name();
                    self.inst_i_eq(i);
                    self.func_2(name, |s| s.write_ref(lhs), |s| s.write_ref(rhs));
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

    fn inst_i(&mut self, i: u32) {
        self.write(&format!("%{i}"));
    }

    fn inst_i_eq(&mut self, i: usize) {
        self.inst_i(i as u32);
        self.write(" = ");
    }

    fn write_ref(&mut self, r: &Ref) {
        match r {
            Ref::Const(ty, i) => self.func_2(
                "const",
                |s| s.write_type_ref(ty),
                |s| s.write(&self.fir.get_const(*i).to_string()),
            ),
            Ref::Inst(i) => self.inst_i(*i),
        }
    }

    fn write_type_ref(&mut self, ty: &TypeRef) {
        use TypeRef::*;
        let str = match ty {
            IntU64 => "u64",
        };
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
    }
}
