use std::collections::HashMap;

use crate::ast::{Ast, DIndex};
use crate::parser::{EIndex, Expr, Binop};

#[derive(Debug, Clone, Copy)]
pub enum ByteCode {
    Nop,
    Push(u64),
    Jump(u32),
    JumpIfZero(u32),
    // (index)
    Store(u32),
    Add,
    Sub,
    Mul,
    Div,
    Eq,
}

impl From<Binop> for ByteCode {
    fn from(op: Binop) -> Self {
        match op {
            Binop::Plus => ByteCode::Add,
            Binop::Minus => ByteCode::Sub,
            Binop::Mul => ByteCode::Mul,
            Binop::Div => ByteCode::Div,
            Binop::Eq => ByteCode::Eq,
        }
    }
}

#[derive(Debug)]
pub struct Chunk {
    pub ops: Vec<ByteCode>,
}

pub struct Compiler {
    ast: Ast,
    bytecode: Chunk,
    expr_i: usize,
    visited: Vec<bool>,
    name_map: HashMap<DIndex, u32>,
}

impl Compiler {
    pub fn new(ast: Ast) -> Self {
        let visited = vec![false; ast.exprs.len()];
        Self {
            ast,
            bytecode: Chunk {
                ops: Vec::new(),
            },
            expr_i: 0,
            visited,
            name_map: HashMap::new(),
        }
    }

    fn next_expr(&mut self) -> Option<Expr> {
        let num_exprs = self.ast.exprs.len();

        // skip visited
        // TODO: is there a way to know what has been visited based on tree structure?
        while self.expr_i < num_exprs && self.visited[self.expr_i] {
            self.expr_i += 1;
        }
        if self.expr_i >= self.ast.exprs.len() {
            return None;
        }
        let expr = self.ast.exprs[self.expr_i];
        self.mark_visited(self.expr_i);
        self.expr_i += 1;
        Some(expr)
    }

    fn mark_visited(&mut self, expr_i: usize) {
        self.visited[expr_i] = true;
        dbg!(&self.visited);
        dbg!(&self.ast.exprs);
    }

    pub fn compile(&mut self) {
        while let Some(expr) = self.next_expr() {
            self._compile_expr(expr);
        }
    }

    fn _compile_expr(&mut self, expr: Expr) {
        match expr {
            Expr::Int(i) => {
                self.compile_int(i);
            }
            Expr::Binop { op, lhs, rhs } => {
                self.compile_binop(op, lhs, rhs);
            }
            Expr::If { cond, branch_true, branch_false } => {
                self.compile_if(cond, branch_true, branch_false);
            }
            Expr::Bind {name, value} => {
                self.compile_bind(name, value);
            }
            _ => unimplemented!("Expr: {:?} not implemented", expr),
        }
    }

    fn compile_expr(&mut self, i: EIndex) {
        self._compile_expr(self.ast.exprs[i]);
        self.mark_visited(i);
    }

    fn compile_binop(&mut self, op: Binop, lhs: EIndex, rhs: EIndex) {
        self.compile_expr(lhs);
        self.compile_expr(rhs);
        // FIXME:
        let bc_op = ByteCode::from(op);
        self.emit(bc_op);
    }

    fn compile_if(&mut self, cond: EIndex, branch_true: EIndex, branch_false: EIndex) {
        self.compile_expr(cond);

        let jmp_true_i = self.init_jmp();
        let jmp_else_i = self.init_jmp();

        self.end_jmp_if_zero(jmp_true_i);
        self.compile_expr(branch_true);

        let jmp_end_i = self.reserve();

        self.end_jmp(jmp_else_i);
        self.compile_expr(branch_false);

        self.end_jmp(jmp_end_i);
    }

    fn compile_bind(&mut self, name: DIndex, value: EIndex) {
        self.compile_expr(value);
        let i = self.ident_i(name);
        self.emit(ByteCode::Store(i));
    }

    fn ident_i(&mut self, name: DIndex) -> u32 {
        let next = self.name_map.len() as u32;
        let val = self.name_map.entry(name).or_insert(next);
        return *val;
    }

    fn set(&mut self, i: usize, bc: ByteCode) {
        self.bytecode.ops[i] = bc;
    }

    fn emit(&mut self, bc: ByteCode) {
        self.bytecode.ops.push(bc);
    }

    fn reserve(&mut self) -> usize {
        let i = self.bytecode.ops.len();
        self.emit(ByteCode::Nop);
        return i;
    }

    fn init_jmp(&mut self) -> usize {
        return self.reserve();
    }

    fn end_jmp_if_zero(&mut self, jmp_i: usize) {
        let len = self.bytecode.ops.len();
        let jump_offset = len as u32;
        self.set(jmp_i, ByteCode::JumpIfZero(jump_offset));
    }

    fn end_jmp(&mut self, jmp_i: usize) {
        let len = self.bytecode.ops.len();
        let jump_offset = len as u32;
        self.set(jmp_i, ByteCode::Jump(jump_offset));
    }

    fn compile_int(&mut self, di: DIndex) {
        let val: u64 = self.ast.get_const(di);
        self.emit(ByteCode::Push(val));
    }

    pub fn bytecode(&self) -> &Chunk {
        &self.bytecode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::tests::assert_matches;

    fn compile(contents: &str) -> Compiler {
        let parser = crate::parser::Parser::new(contents);
        let mut compiler = Compiler::new(parser.parse().unwrap());
        compiler.compile();
        return compiler;
    }

    macro_rules! assert_bytecode_matches {
        ($bytecode:expr, [$($ops:pat),*]) => {
            let mut i = 0;
            $(
                #[allow(unused_assignments)]
                {
                    assert_matches!($bytecode.ops[i], $ops);
                    i += 1;
                }
            )*
            assert_eq!($bytecode.ops.len(), i, "expected {} ops, got {}. Extra: {:?}", i, $bytecode.ops.len(), &$bytecode.ops[i..]);
        };
    }

    macro_rules! assert_compiles_to {
        ($contents:literal, [$($ops:pat),*]) => {
            let contents = $contents;
            let compiler = compile(contents);
            assert_bytecode_matches!(compiler.bytecode, [$($ops),*]);
        };
    }

    #[test]
    fn add() {
        let contents = "( + 1 2 )";
        let compiler = compile(contents);
        assert_bytecode_matches!(
            compiler.bytecode,
            [ByteCode::Push(1), ByteCode::Push(2), ByteCode::Add]
        );
    }

    #[test]
    fn nested_add() {
        let contents = "( + 1 ( + 2 3 ) )";
        let compiler = compile(contents);
        assert_bytecode_matches!(
            compiler.bytecode,
            [
                ByteCode::Push(1),
                ByteCode::Push(2),
                ByteCode::Push(3),
                ByteCode::Add,
                ByteCode::Add
            ]
        );
    }

    #[test]
    fn repeated_add() {
        let contents = "(+ 1 2) (+ 3 4)";
        let compiler = compile(contents);
        assert_bytecode_matches!(
            compiler.bytecode,
            [
                ByteCode::Push(1),
                ByteCode::Push(2),
                ByteCode::Add,
                ByteCode::Push(3),
                ByteCode::Push(4),
                ByteCode::Add
            ]
        );
    }

    #[test]
    fn all_binops() {
        let contents = "(+ 1 2) (- 3 4) (* 5 6) (/ 7 8)";
        let compiler = compile(contents);
        assert_bytecode_matches!(
            compiler.bytecode,
            [
                ByteCode::Push(1),
                ByteCode::Push(2),
                ByteCode::Add,
                ByteCode::Push(3),
                ByteCode::Push(4),
                ByteCode::Sub,
                ByteCode::Push(5),
                ByteCode::Push(6),
                ByteCode::Mul,
                ByteCode::Push(7),
                ByteCode::Push(8),
                ByteCode::Div
            ]
        );
    }

    #[test]
    fn if_expr() {
        let contents = "(if 1 2 3)";
        let compiler = compile(contents);
        dbg!(&compiler.bytecode.ops);
        assert_bytecode_matches!(
            compiler.bytecode,
            [
                ByteCode::Push(1),
                ByteCode::JumpIfZero(3),
                ByteCode::Jump(5),
                ByteCode::Push(2),
                ByteCode::Jump(6),
                ByteCode::Push(3)
            ]
        );
    }

    #[test]
    fn let_bind() {
        assert_compiles_to!(
            "(let x 1)",
            [
                ByteCode::Push(1),
                ByteCode::Store(0)
            ]
        );
    }
}
