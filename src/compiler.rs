use crate::ast::Ast;
use crate::lexer::Token;
use crate::parser::{Expr, TIndex};

#[derive(Debug, Clone, Copy)]
pub enum ByteCode {
    Push(u64),
    Add,
}

pub struct Compiler<'a> {
    ast: Ast<'a>,
    bytecode: Vec<ByteCode>,
    expr_i: usize,
    visited: Vec<bool>,
}

impl<'a> Compiler<'a> {
    pub fn new(ast: Ast<'a>) -> Self {
        let visited = vec![false; ast.exprs.len()];
        Self {
            ast,
            bytecode: Vec::new(),
            expr_i: 0,
            visited,
        }
    }

    fn next_expr(&mut self) -> Option<Expr> {
        let num_exprs = self.ast.exprs.len();

        // skip visited
        while self.expr_i < num_exprs && self.visited[self.expr_i] {
            self.expr_i += 1;
        }
        if self.expr_i >= self.ast.exprs.len() {
            return None;
        }
        let expr = self.ast.exprs[self.expr_i];
        self.visit(self.expr_i);
        self.expr_i += 1;
        Some(expr)
    }

    fn visit(&mut self, expr_i: usize) {
        self.visited[expr_i] = true;
    }

    pub fn compile(&mut self) {
        while let Some(expr) = self.next_expr() {
            match expr {
                Expr::Int(i) => {
                    self.compile_int(i);
                }
                Expr::Binop { op, lhs, rhs } => {
                    self.visit(lhs);
                    self.visit(rhs);
                    self.bytecode.push(ByteCode::Push(lhs as u64));
                    self.bytecode.push(ByteCode::Push(rhs as u64));
                    match self.ast.tokens[op] {
                        Token::Plus => {
                            self.bytecode.push(ByteCode::Add);
                        }
                        _ => unimplemented!(),
                    }
                }
                _ => unimplemented!(),
            }
        }
    }

    fn push(&mut self, bc: ByteCode) {
        self.bytecode.push(bc);
    }

    fn parse_int(&self, tok_i: TIndex) -> u64 {
        let val = self.ast.token_slice(tok_i);
        return val.parse().unwrap();
    }

    fn compile_int(&mut self, tok_i: TIndex) {
        let val = self.parse_int(tok_i);
        self.push(ByteCode::Push(val));
    }

    pub fn bytecode(&self) -> &Vec<ByteCode> {
        &self.bytecode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::tests::{assert_matches, let_assert_matches};

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
                    assert_matches!($bytecode[i], $ops);
                    i += 1;
                }
            )*
            assert_eq!($bytecode.len(), i, "expected {} ops, got {}", i, $bytecode.len());
        };
    }

    #[test]
    fn add() {
        let contents = "( + 1 2 )";
        let compiler = compile(contents);
        assert_bytecode_matches!(compiler.bytecode, [
            ByteCode::Push(1),
            ByteCode::Push(2),
            ByteCode::Add
        ]);
    }
}
