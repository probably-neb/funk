use crate::lexer;
use crate::parser;

pub struct Ast {
    pub exprs: Vec<parser::Expr>,
    pub tokens: Vec<lexer::Token>,
}
