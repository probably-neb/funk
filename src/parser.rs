use anyhow::{anyhow, Result};

use crate::{lexer::{Token, Lexer}, ast::Ast};

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

#[allow(dead_code)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Expr {
    Nop,
    Int(TIndex),
    Ident(TIndex),
    If {
        cond: EIndex,
        branch_true: EIndex,
        branch_false: EIndex,
    },
    Binop {
        op: TIndex,
        lhs: EIndex,
        rhs: EIndex,
    },
    String(TIndex),
    FunDef {
        name: TIndex,
        args: EIndex,
        body: EIndex,
    },
    FunArg {
        name: TIndex,
        len: u8,
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

impl<'a> Parser<'a> {
    fn _new(lxr: Lexer<'a>) -> Parser<'a> {
        Parser {
            lxr,
            tokens: Vec::new(),
            exprs: Vec::new(),
            tok_i: 0,
        }
    }
    pub fn new(contents: &'a str) -> Parser<'a> {
        Parser::_new(Lexer::new(contents))
    }

    pub fn parse(mut self) -> Result<crate::ast::Ast> {
        self._parse()?;
        Ok(Ast {
            exprs: self.exprs,
            tokens: self.tokens,
        })
    }

    fn _parse(&mut self) -> Result<()> {
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
                self.tok_i = self.tokens.len() - 1;
                return Some(tok);
            }
        }
    }

    fn push(&mut self, expr: Expr) -> EIndex {
        self.exprs.push(expr);
        return self.exprs.len() - 1;
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
            Token::Ident(_) => Ok(Expr::Ident(self.tok_i)),
            Token::String(_) => Ok(Expr::String(self.tok_i)),
            Token::If => return Some(self.if_expr()),
            Token::Fun => return Some(self.fun_expr()),
            Token::Eq | Token::Mul | Token::Plus | Token::Minus => {
                return Some(self.binop_expr(self.tok_i))
            }
            _ => unimplemented!("{:?} not implemented", tok),
        };

        match expr {
            Err(err) => return Some(Err(err)),
            Ok(expr) => return Some(Ok(self.push(expr))),
        }
    }

    fn binop_expr(&mut self, op: TIndex) -> Result<EIndex> {
        let expr_i = self.reserve();
        let lhs = self.expr().unwrap()?;
        let rhs = self.expr().unwrap()?;
        eat!(self, Token::RParen)?;
        self.exprs[expr_i] = Expr::Binop { op, lhs, rhs };
        Ok(expr_i)
    }

    fn if_expr(&mut self) -> Result<EIndex> {
        let if_i = self.reserve();
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

    fn fun_expr(&mut self) -> Result<EIndex> {
        let fun_i = self.reserve();
        eat!(self, Token::Ident(_))?;
        let name = self.tok_i;
        let args = self.fun_args()?;
        let body = self.expr().unwrap()?;
        eat!(self, Token::RParen)?;
        self.exprs[fun_i] = Expr::FunDef { name, args, body };
        Ok(fun_i)
    }

    fn fun_args(&mut self) -> Result<EIndex> {
        let mut num = 0_u8;
        eat!(self, Token::LParen)?;
        let (first, first_name) = match self.tok() {
            Some(Token::RParen) => {
                // FIXME: return None for no args
                let emtpy_arg = self.push(Expr::FunArg { name: 0, len: 0 });
                return Ok(emtpy_arg);
            }
            Some(Token::Ident(_)) => (self.reserve(), self.tok_i),
            _ => return Err(anyhow!("incomplete args")),
        };
        while let Some(Token::Ident(_)) = self.tok() {
            num += 1;
            self.push(Expr::FunArg {
                name: self.tok_i,
                len: num,
            });
        }
        self.exprs[first] = Expr::FunArg {
            name: first_name,
            len: num,
        };
        assert_eq!(self.tokens[self.tok_i], Token::RParen);
        Ok(first)
    }
}

// NOTE: something is detecting variables declared in test macros as unused
// this is a bug in clippy / the lsp I believe, the lsp picks up on
// the fact the variables are the same for renaming + undeclared var
// warning and the code works fine
// hence the allow unused variables
#[allow(unused_variables)]
#[cfg(test)]
mod tests {
    use super::*;

    fn parse<'a>(contents: &'a str) -> Result<Parser<'a>> {
        let mut parser = Parser::new(contents);
        parser._parse()?;
        Ok(parser)
    }


    #[cfg(test)]
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
                _ => unreachable!(),
            }
        }};
    }

    macro_rules! let_assert_matches {
        ($expr:expr, $pat:pat, $message:literal, $($arg:tt)*) => {
            assert!(matches!($expr, $pat), $message, $($arg)*);
            let $pat = $expr else { unreachable!() };
        };
        ($expr:expr, $pat:pat) => {
            assert!(matches!($expr, $pat), "expected {:?}, got {:?}", stringify!($pat), $expr);
            let $pat = $expr else { unreachable!() };
        };
    }

    macro_rules! assert_tok_is {
        ($parser:expr, $i:ident, $tok:pat) => {
            assert_matches!($parser.tokens[$i], $tok)
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
        assert_eq!( parser.exprs[0], Expr::Binop { op: 1, lhs: 1, rhs: 2 });
    }

    #[test]
    fn eq_with_sub_expr() {
        let contents = "(= (* 2 2) 4)";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(
            parser.exprs[0],
            Expr::Binop {
                op,
                lhs: 1,
                rhs: 4
            }
        );
        assert_tok_is!(parser, op, Token::Eq);

        let_assert_matches!(
            parser.exprs[1],
            Expr::Binop {
                op,
                lhs: 2,
                rhs: 3
            }
        );
        assert_tok_is!(parser, op, Token::Mul);
    }

    #[test]
    fn mul_expr() {
        let contents = "(* 10 10)";
        let parser = parse(contents).expect("parser error");
        assert_eq!(parser.tokens.len(), 5);
        let_assert_matches!(parser.exprs[0], Expr::Binop{op, lhs: 1, rhs: 2});
        assert_tok_is!(parser, op, Token::Mul);
    }

    #[test]
    fn if_expr() {
        let contents = r#"(if (= 4 4) "yes" "no")"#;
        let parser = parse(contents).expect("parser error");

        let_assert_matches!(parser.exprs[0], Expr::If {
                cond,
                branch_true,
                branch_false,
            }
        );
        assert_matches!(parser.exprs[cond], Expr::Binop { op: _, lhs: _, rhs: _ });
        assert_matches!(parser.exprs[branch_true], Expr::String(_));
        assert_matches!(parser.exprs[branch_false], Expr::String(_));
    }

    #[test]
    fn fun_args() {
        let contents = "(a b)";
        let mut parser = Parser::new(contents);
        let args = parser.fun_args().expect("parser error");
        assert_eq!(args, 0);
        assert_matches!(
            parser.exprs[0],
            Expr::FunArg {
                name: _name,
                len: 1
            }
        );
        assert_matches!(parser.exprs[1], Expr::FunArg { name: 2, len: 1 });
    }

    #[test]
    fn fun_expr() {
        let contents = "(fun foo (a b) (+ a b))";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(
            parser.exprs[0],
            Expr::FunDef {
                name,
                args,
                body
            }
        );
        let_assert_matches!(
            parser.tokens[name],
            Token::Ident(range)
        );
        assert_eq!(parser.lxr.as_str(&range), "foo");
        assert_matches!( parser.exprs[body], Expr::Binop { op: _, lhs: _, rhs: _ });
    }
}
