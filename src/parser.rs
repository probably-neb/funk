use anyhow::{anyhow, Result};

use crate::{lexer::Token, Lexer};

/// Expression index
type EIndex = usize;

/// Token index
type TIndex = usize;

pub struct Parser<'a> {
    lxr: Lexer<'a>,
    tokens: Vec<Token>,
    tok_i: EIndex,
    exprs: Vec<Expr>,
}

#[derive(Debug, PartialEq)]
pub enum Expr {
    Nop,
    Seq(Vec<EIndex>),
    Int(TIndex),
    If {
        cond: EIndex,
        branch_true: EIndex,
        branch_false: EIndex,
    },
    Binop {
        op: Token,
        lhs: EIndex,
        rhs: EIndex,
    },
}

macro_rules! eat {
    ($self:ident, $tok:pat) => {
        match $self.tok() {
            tok @ Some($tok) => Ok(tok),
            Some(tok) => Err(anyhow!("expected {:?}, got {:?}", stringify!($tok), tok)),
            None => Err(anyhow!("expected {:?}, got EOF", stringify!($tok))),
        }
    };
}

pub fn parse<'a>(contents: &'a str) -> Result<Parser<'a>> {
    let mut parser = Parser::new(Lexer::new(contents));
    parser.parse()?;
    Ok(parser)
}

impl<'a> Parser<'a> {
    pub fn new(lxr: Lexer<'a>) -> Parser<'a> {
        Parser {
            lxr,
            tokens: Vec::new(),
            exprs: Vec::new(),
            tok_i: 0,
        }
    }

    pub fn parse(&mut self) -> Result<()> {
        while let Some(expr) = self.expr() {
            expr?;
        }
        Ok(())
    }

    fn tok(&mut self) -> Option<Token> {
        let tok = self.lxr.next_token().expect("lexer error");
        match tok {
            Token::Eof => None,
            _ => {
                self.tokens.push(tok);
                if self.tok_i == 0 && self.tokens.len() == 1 {
                    return Some(self.tokens[0]);
                }
                self.tok_i += 1;
                return Some(self.tokens[self.tok_i]);
            }
        }
    }

    fn push(&mut self, expr: Expr) -> EIndex {
        self.exprs.push(expr);
        self.exprs.len() - 1
    }

    fn reserve(&mut self) -> EIndex {
        return self.push(Expr::Nop);
    }

    fn expr(&mut self) -> Option<Result<EIndex>> {
        let mut tok = self.tok()?;
        if tok == Token::LParen {
            tok = self.tok()?;
        }
        let expr = match tok {
            Token::Int(_) => Ok(Expr::Int(self.tok_i)),
            Token::If => return Some(self.if_expr()),
            Token::Eq | Token::Mul | Token::Plus | Token::Minus => {
                return Some(self.binop_expr(tok))
            }
            _ => unimplemented!(),
        };

        match expr {
            Err(err) => return Some(Err(err)),
            Ok(expr) => return Some(Ok(self.push(expr))),
        }
    }

    fn binop_expr(&mut self, op: Token) -> Result<EIndex> {
        let eq_i = self.reserve();
        let lhs = self.expr().unwrap()?;
        let rhs = self.expr().unwrap()?;
        eat!(self, Token::RParen)?;
        self.exprs[eq_i] = Expr::Binop { op, lhs, rhs };
        Ok(eq_i)
    }

    fn if_expr(&mut self) -> Result<EIndex> {
        eat!(self, Token::If)?;
        let if_i = self.push(Expr::If {
            cond: 0,
            branch_true: 0,
            branch_false: 0,
        });
        let cond = self.expr().unwrap()?;
        let branch_true = self.expr().unwrap()?;
        let branch_false = self.expr().unwrap()?;
        eat!(self, Token::RParen)?;
        self.exprs[if_i] = Expr::If {
            cond,
            branch_true,
            branch_false,
        };
        Ok(if_i)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! assert_matches {
        ($expr:expr, $pat:pat) => {
            assert!(matches!($expr, $pat))
        };
        ($expr:expr, $pat:pat, $message:literal) => {
            assert!(matches!($expr, $pat if $cond), $message)
        };
    }

    #[test]
    fn literal() {
        let contents = "10";
        let parser = parse(contents).expect("parser error");
        assert_eq!(parser.tokens.len(), 1);
        assert_eq!(parser.tokens[0], Token::Int((0, 2)));
        assert_eq!(parser.exprs.len(), 1);
        assert_eq!(parser.exprs[0], Expr::Int(0));
    }

    #[test]
    fn eq_expr() {
        let contents = "(= 10 10)";
        let parser = parse(contents).expect("parser error");
        assert_eq!(parser.tokens.len(), 5);
        assert_eq!(
            parser.exprs[0],
            Expr::Binop {
                op: Token::Eq,
                lhs: 1,
                rhs: 2
            }
        );
    }

    #[test]
    fn eq_with_sub_expr() {
        let contents = "(= (* 2 2) 4)";
        let parser = parse(contents).expect("parser error");
        assert_matches!(
            parser.exprs[0],
            Expr::Binop {
                op: Token::Eq,
                lhs: 1,
                rhs: 4
            }
        );
        assert_matches!(
            parser.exprs[1],
            Expr::Binop {
                op: Token::Mul,
                lhs: 2,
                rhs: 3
            }
        )
    }

    #[test]
    fn mul_expr() {
        let contents = "(* 10 10)";
        let parser = parse(contents).expect("parser error");
        assert_eq!(parser.tokens.len(), 5);
        assert_eq!(
            parser.exprs[0],
            Expr::Binop {
                op: Token::Mul,
                lhs: 1,
                rhs: 2
            }
        );
    }

    #[test]
    fn if_expr() {
        let contents = r#"(if (= (* 2 2) 4) "yes" "no")"#;
        let parser = parse(contents).expect("parser error");
        assert_eq!(parser.tokens.len(), 13);
        // assert_eq!(parser.tokens[0], Token::Int((0, 2)));
        // assert_eq!(parser.exprs.len(), 1);
        // assert_eq!(parser.exprs[0], Expr::Int(0));
    }
}
