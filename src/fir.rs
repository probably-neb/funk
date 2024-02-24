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

pub struct FIR {
    ops: Vec<Op>,
    extra: Extra,
    data: ast::DataPool,
}

impl FIR {
    fn new(ops: Vec<Op>, extra: Extra, data: ast::DataPool) -> Self {
        return Self { ops, extra, data };
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
                return Ok(self.push(Op::Load(Ref::Const(i))));
            }
            _ => unimplemented!("unimplemented expr {:?}", expr),
        }
    }

    fn gen_binop(&mut self, op: crate::parser::Binop, lhs: usize, rhs: usize) -> Result<usize> {
        let lhs = self.gen_expr(lhs)?;
        let rhs = self.gen_expr(rhs)?;

        use crate::parser::Binop::*;

        let i = match op {
            Add => self.push(Op::Add(Ref::Inst(lhs as u32), Ref::Inst(rhs as u32))),

            _ => unimplemented!("unimplemented binop {:?}", op),
        };
        Ok(i)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Ref {
    Inst(u32),
    Const(u32),
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
                Load(Ref::Const(0)),
                Load(Ref::Const(1)),
                Add(Ref::Inst(0), Ref::Inst(1))
            ]
        );
    }
}
