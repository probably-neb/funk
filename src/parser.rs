use anyhow::{anyhow, Result};

use crate::{
    ast::{Ast, DIndex, DataPool},
    lexer::{self, Lexer, Token},
};

/// Expression index
pub type EIndex = usize;

/// Token index
pub type TIndex = usize;

pub struct Parser<'a> {
    lxr: Lexer<'a>,
    tokens: Vec<Token>,
    tok_i: EIndex,
    exprs: Vec<Expr>,
    data: DataPool,
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Expr {
    Nop,
    Int(DIndex),
    Ident(DIndex),
    String(DIndex),
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
    FunDef {
        name: DIndex,
        args: EIndex,
        body: EIndex,
    },
    FunArg {
        name: DIndex,
        len: u8,
    },
}

macro_rules! eat {
    ($self:ident, $tok:pat) => {
        match $self.tok() {
            // binds tok to the value of the token within Some
            Some(tok @ $tok) => Ok(tok),
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
            data: DataPool::new(),
        }
    }
    pub fn new(contents: &'a str) -> Parser<'a> {
        Parser::_new(Lexer::new(contents))
    }

    pub fn parse(mut self) -> Result<crate::ast::Ast> {
        self._parse()?;
        Ok(Ast::new(self.exprs, self.data))
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

    // because AST is linearized reserve allows pushing a placeholder
    // while the information required for the head of a subtree is gathered
    // i.e. the index of the lhs and rhs of a binop
    fn reserve(&mut self) -> EIndex {
        return self.push(Expr::Nop);
    }

    fn expr(&mut self) -> Option<Result<EIndex>> {
        let mut tok = self.tok()?;
        if tok == Token::LParen {
            tok = self.tok()?;
        }
        let expr = match tok {
            Token::Int(range) => Ok(Expr::Int(self.intern_int(range))),
            Token::Ident(range) => Ok(Expr::Ident(self.intern_str(range))),
            Token::String(range) => Ok(Expr::String(self.intern_str(range))),
            Token::If => return Some(self.if_expr()),
            Token::Fun => return Some(self.fun_expr()),
            Token::Eq | Token::Mul | Token::Plus | Token::Minus | Token::Div => {
                return Some(self.binop_expr(self.tok_i))
            }
            _ => unimplemented!("{:?} not implemented", tok),
        };

        match expr {
            Err(err) => return Some(Err(err)),
            Ok(expr) => return Some(Ok(self.push(expr))),
        }
    }

    fn intern_int(&mut self, range: lexer::Range) -> DIndex {
        let bytes = self.lxr.slice(&range);
        let val = unsafe { std::str::from_utf8_unchecked(bytes) }
            // TODO: support alternative int sizes
            .parse::<u64>()
            .expect("invalid int");
        return self.data.put(&val);
    }

    fn intern_str(&mut self, range: lexer::Range) -> DIndex {
        let str = self.lxr.slice(&range);
        return self.data.put(str);
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
        let Token::Ident(range) = eat!(self, Token::Ident(_))? else {
            unreachable!()
        };
        let name = self.intern_str(range);
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
            Some(Token::Ident(range)) => (self.reserve(), self.intern_str(range)),
            _ => return Err(anyhow!("incomplete args")),
        };
        while let Some(Token::Ident(range)) = self.tok() {
            num += 1;
            let name = self.intern_str(range);
            self.push(Expr::FunArg { name, len: num });
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
pub mod tests {
    use super::*;

    pub fn parse<'a>(contents: &'a str) -> Result<Parser<'a>> {
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

    pub(crate) use assert_matches;

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
    pub(crate) use let_assert_matches;

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
        assert_eq!(
            parser.exprs[0],
            Expr::Binop {
                op: 1,
                lhs: 1,
                rhs: 2
            }
        );
    }

    #[test]
    fn eq_with_sub_expr() {
        let contents = "(= (* 2 2) 4)";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::Binop { op, lhs: 1, rhs: 4 });
        assert_tok_is!(parser, op, Token::Eq);

        let_assert_matches!(parser.exprs[1], Expr::Binop { op, lhs: 2, rhs: 3 });
        assert_tok_is!(parser, op, Token::Mul);
    }

    #[test]
    fn mul_expr() {
        let contents = "(* 10 10)";
        let parser = parse(contents).expect("parser error");
        assert_eq!(parser.tokens.len(), 5);
        let_assert_matches!(parser.exprs[0], Expr::Binop { op, lhs: 1, rhs: 2 });
        assert_tok_is!(parser, op, Token::Mul);
    }

    #[test]
    fn if_expr() {
        let contents = r#"(if (= 4 4) "yes" "no")"#;
        let parser = parse(contents).expect("parser error");

        let_assert_matches!(
            parser.exprs[0],
            Expr::If {
                cond,
                branch_true,
                branch_false,
            }
        );
        assert_matches!(
            parser.exprs[cond],
            Expr::Binop {
                op: _,
                lhs: _,
                rhs: _
            }
        );
        assert_matches!(parser.exprs[branch_true], Expr::String(_));
        assert_matches!(parser.exprs[branch_false], Expr::String(_));
    }

    #[test]
    fn fun_args() {
        let contents = "(a b)";
        let mut parser = Parser::new(contents);
        let args = parser.fun_args().expect("parser error");
        assert_eq!(args, 0);
        assert_matches!(parser.exprs[0], Expr::FunArg { name, len: 1 });
        assert_matches!(parser.exprs[1], Expr::FunArg { name, len: 1 });
    }

    #[test]
    fn fun_expr() {
        let contents = "(fun foo (a b) (+ a b))";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunDef { name, args, body });
        assert_matches!(
            parser.exprs[body],
            Expr::Binop {
                op: _,
                lhs: _,
                rhs: _
            }
        );
    }

    #[test]
    fn ident_interned() {
        let contents = "(+ a b)";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::Binop { op, lhs, rhs });
        let_assert_matches!(parser.exprs[lhs], Expr::Ident(a));
        let_assert_matches!(parser.exprs[rhs], Expr::Ident(b));
        assert_eq!(parser.data.get_ref::<str>(a), "a");
        assert_eq!(parser.data.get_ref::<str>(b), "b");
    }

    #[test]
    fn fun_name_interned() {
        let contents = "(fun foo (foo bar) (+ foo bar))";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunDef { name, args, body });
        assert_eq!(parser.data.get_ref::<str>(name), "foo");
    }

    #[test]
    fn fun_arg_interned() {
        let contents = "(fun foo (foo bar) (+ foo bar))";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunDef { name, args, body });
        let_assert_matches!(parser.exprs[args], Expr::FunArg { name: foo, len: 1 });
        let_assert_matches!(parser.exprs[args + 1], Expr::FunArg { name: bar, len: 1 });
        assert_eq!(parser.data.get_ref::<str>(foo), "foo");
        assert_eq!(parser.data.get_ref::<str>(bar), "bar");
    }
}
