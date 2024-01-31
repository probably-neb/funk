use crate::lexer;
use crate::parser;

pub struct Ast<'a> {
    pub exprs: Vec<parser::Expr>,
    pub tokens: Vec<lexer::Token>,
    src: &'a [u8],
}

impl<'a> Ast<'a> {
    pub fn new(exprs: Vec<parser::Expr>, tokens: Vec<lexer::Token>, src: &'a [u8]) -> Self {
        Self {
            exprs,
            tokens,
            src,
        }
    }

    fn slice(&self, range: (usize, usize)) -> &'a [u8] {
        let start = range.0;
        let end = range.1.min(self.src.len());
        return &self.src[start..end];
    }
    pub fn token_slice(&self, tok_i: usize) -> &'a str {
        let tok = self.tokens[tok_i];
        let range =  match tok {
            lexer::Token::Ident(range) => range,
            lexer::Token::Int(range) => range,
            lexer::Token::Float(range) => range,
            lexer::Token::String(range) => range,
            _ => unreachable!("tried to get range of non-range token: {:?}", stringify!(tok))
        };
        let slice = self.slice(range);
        return std::str::from_utf8(slice).unwrap();
    }
}
