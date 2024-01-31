use crate::ast::Ast;
use crate::lexer::Token;
use crate::parser::Expr;

#[derive(Debug, Clone, Copy)]
pub enum ByteCode {
    Push(u64),
    Add
}

pub struct Compiler {
    ast: Ast,
    bytecode: Vec<ByteCode>
}

impl Compiler {
    pub fn new(ast: Ast) -> Self {
        Self {
            ast,
            bytecode: Vec::new()
        }
    }
    pub fn compile(&mut self) {
        for expr in self.ast.exprs.iter() {
            match expr {
                Expr::Int(i) => {
                    self.bytecode.push(ByteCode::Push(*i as u64));
                },
                Expr::Binop { op, lhs, rhs } => {
                    self.bytecode.push(ByteCode::Push(*lhs as u64));
                    self.bytecode.push(ByteCode::Push(*rhs as u64));
                    match self.ast.tokens[*op] {
                        Token::Plus => {
                            self.bytecode.push(ByteCode::Add);
                        },
                        _ => unimplemented!()
                    }
                },
                _ => unimplemented!()
            }
        }
    }

    pub fn bytecode(&self) -> &Vec<ByteCode> {
        &self.bytecode
    }
}
