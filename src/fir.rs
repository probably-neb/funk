use anyhow::{anyhow, Context, Result};

pub mod stringify;
pub use crate::ast::DIndex;
use crate::ast::{self, Ast, Extra};
use crate::parser::Expr;
use stringify::stringify;

/// An index into the `extra` field of `FIR`
pub type XIndex = usize;

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Op {
    /* Fucntions */
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
        /// a ref to the FunDef op
        fun: u32,
        /// a pointer to a slice of Refs in extra
        args: XIndex,
    },
    Ret(Ref),

    /* Memory */
    /// declare a local variable
    /// NOTE: these should be hoisted to the top of the function body
    /// for eaasier codegen
    Alloc,
    Load(Ref),
    /// (dest, src)
    Store(Ref, Ref),

    /* Binops */
    Add(Ref, Ref),
    Sub(Ref, Ref),
    Mul(Ref, Ref),
    Div(Ref, Ref),
    Eq(Ref, Ref),
    Lt(Ref, Ref),
    LtEq(Ref, Ref),
    Gt(Ref, Ref),
    GtEq(Ref, Ref),

    /* Control flow */
    Label,
    Branch {
        cond: Ref,
        t: Ref,
        f: Ref,
    },
    Phi {
        a: (Ref, Ref),
        b: (Ref, Ref),
    },
    Jump(Ref),
}

impl Op {
    pub fn is_ctrl_flow(&self) -> bool {
        match self {
            Op::Branch { .. } | Op::Jump(_) | Op::Ret(_) => true,
            _ => false
        }
    }
}

pub struct FIR {
    pub ops: Vec<Op>,
    pub types: Vec<TypeRef>,
    pub extra: Extra,
    pub data: ast::DataPool,
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

    // TODO: take type as param
    pub fn get_const(&self, i: u32) -> u64 {
        let di = self.extra[i as usize];
        return self.data.get::<u64>(di as usize);
    }

    pub fn stringify(&self) -> String {
        return crate::fir::stringify::stringify(self);
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
                    Div => Op::Div(lhs, rhs),
                    Eq => Op::Eq(lhs, rhs),
                    Lt => Op::Lt(lhs, rhs),
                    Gt => Op::Gt(lhs, rhs),
                    _ => unimplemented!("unimplemented binop {:?}", op),
                };
                let op_ty = self.join_types(lhs_i, rhs_i);

                let i = self.push_typed(op_inst, op_ty);
                Ok(i)
            }
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

                let fun_op = Op::FunCall {
                    fun: fun_i,
                    args: args_i,
                };
                let ret_ty = self.types[fun_i as usize];
                let res = self.push_typed(fun_op, ret_ty);
                return Ok(res);
            }
            Expr::If {
                cond,
                branch_true,
                branch_false,
            } => {
                let cond_i = self.gen_expr(cond)?;
                let cond_ref = Ref::Inst(cond_i as u32);
                let br_i = self.reserve();

                let true_i = self.begin_basic_block();
                let true_res_i = self.gen_expr(branch_true)?;
                let true_end_i = self.reserve();
                let true_ref = Ref::Inst(true_i as u32);
                let true_res_ref = Ref::Inst(true_res_i as u32);

                let false_i = self.begin_basic_block();
                let false_res_i = self.gen_expr(branch_false)?;
                let false_end_i = self.reserve();
                let false_ref = Ref::Inst(false_i as u32);
                let false_res_ref = Ref::Inst(false_res_i as u32);

                let branch_op = Op::Branch {
                    cond: cond_ref,
                    t: true_ref,
                    f: false_ref,
                };
                self.set(br_i, branch_op);

                let end_i = self.begin_basic_block();
                let jmp_end_op = Op::Jump(Ref::Inst(end_i as u32));
                self.set(true_end_i, jmp_end_op);
                self.set(false_end_i, jmp_end_op);

                let res_op = Op::Phi {
                    a: (true_ref, true_res_ref),
                    b: (false_ref, false_res_ref),
                };
                let res_ty = self.join_types(true_res_i, false_res_i);
                let res_i = self.push_typed(res_op, res_ty);

                return Ok(res_i);
            }
            _ => unimplemented!("unimplemented expr {:?}", expr),
        }
    }

    fn begin_basic_block(&mut self) -> usize {
        return self.push(Op::Label);
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
                    fun: 0,
                    args: 2
                }
            ]
        );
    }


    #[test]
    fn echo() {
        let contents = r#"(fun echo (a) a)"#;
        use Op::*;
        assert_results_in_fir!(
            contents,
            [
                FunDef {..},
                FunArg,
                Load(Ref::Inst(0)),
                Ret(Ref::Inst(1)),
                FunEnd
            ]
        );
    }
}
