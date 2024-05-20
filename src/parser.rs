use anyhow::{anyhow, Context, Result};

use crate::{
    ast::{self, Ast, DIndex, DataPool, Range},
    lexer::{self, Lexer, Token},
};
pub use ast::Expr;

/// Expression index into `exprs`
pub type EIndex = usize;

/// Token index into `tokens`
pub type TIndex = usize;

/// index into `extra`
pub type XIndex = usize;

const DEFAULT_ARGS_CAP: usize = 16;

#[derive(Clone)]
pub struct Parser<'a> {
    lxr: Lexer<'a>,
    tok_i: TIndex,
    tokens: Vec<Token>,
    exprs: Vec<Expr>,
    data: DataPool,
    extra: ast::Extra,
    types: Vec<ast::Type>,
}

macro_rules! eat {
    ($self:ident, $tok:pat) => {
        match $self.next_tok() {
            // binds tok to the value of the token within Some
            Some(tok @ $tok) => Ok(tok),
            Some(tok) => Err(anyhow!("expected {:?}, got {:?}", stringify!($tok), tok)),
            None => Err(anyhow!("expected {:?}, got EOF", stringify!($tok))),
        }
    };
    ($self:ident, $tok:pat, $msg:literal) => {
        match $self.next_tok() {
            // binds tok to the value of the token within Some
            Some(tok @ $tok) => Ok(tok),
            Some(tok) => Err(anyhow!(
                "expected {:?}, got {:?} at {:?}",
                stringify!($tok),
                tok,
                $msg
            )),
            None => Err(anyhow!(
                "expected {:?}, got EOF at {:?}",
                stringify!($tok),
                $msg
            )),
        }
    };
}

impl<'a> Parser<'a> {
    fn _new(lxr: Lexer<'a>) -> Parser<'a> {
        Parser {
            lxr,
            tokens: Vec::new(),
            exprs: Vec::new(),
            tok_i: usize::MAX,
            data: DataPool::new(),
            extra: ast::Extra::new(),
            types: Vec::new(),
        }
    }
    pub fn new(contents: &'a str) -> Parser<'a> {
        Parser::_new(Lexer::new(contents))
    }

    pub fn ast(self) -> Ast {
        Ast::new(self.exprs, self.data, self.extra, self.types)
    }
    pub fn parse(mut self) -> Result<crate::ast::Ast> {
        self._parse()?;
        Ok(self.ast())
    }

    fn _parse(&mut self) -> Result<()> {
        while let Some(expr) = self.expr() {
            expr?;
        }
        Ok(())
    }

    fn peeked_first_tok(&self) -> bool {
        return self.tok_i == usize::MAX && !self.tokens.is_empty();
    }

    /// common logic between `tok()` and `peek()`
    /// returns ({next index to peek}, {has peeked})
    fn _peek_i(&self) -> (usize, bool) {
        // tok_i ?= usize::MAX -> 0
        // else -> tok_i + 1
        let peek_i = self.tok_i.wrapping_add(1);
        let has_peeked = peek_i < self.tokens.len();
        return (peek_i, has_peeked);
    }

    fn tok(&self) -> Option<Token> {
        self.tokens.get(self.tok_i).copied()
    }

    fn next_tok(&mut self) -> Option<Token> {
        // peeked first token
        if self.peeked_first_tok() {
            self.tok_i = 0;
            return Some(self.tokens[0]);
        }
        if self.tok_i == usize::MAX && !self.tokens.is_empty() {
            panic!("usize::MAX tokens reached");
        }
        let (peek_i, has_peeked) = self._peek_i();

        // advance tok_i regardless of whether
        // the next token has been peeked or not
        self.tok_i = peek_i;

        if has_peeked {
            let tok = self.tokens[self.tok_i];
            return Some(tok);
        }

        let tok = self.lxr.next_token().expect("lexer error");
        if tok == Token::Eof {
            self.tok_i = self.tokens.len();
            return None;
        }
        self.tokens.push(tok);
        return Some(tok);
    }

    fn peek_tok(&mut self) -> Option<Token> {
        let (peek_i, has_peeked) = self._peek_i();
        if has_peeked {
            return Some(self.tokens[peek_i]);
        }
        let tok = self.lxr.next_token().expect("lexer error");
        match tok {
            Token::Eof => None,
            _ => {
                self.tokens.push(tok);
                return Some(tok);
            }
        }
    }

    fn push(&mut self, expr: Expr) -> EIndex {
        self.exprs.push(expr);
        self.types.push(ast::Type::Unknown);
        return self.exprs.len() - 1;
    }

    fn push_typed(&mut self, expr: Expr, ty: ast::Type) -> EIndex {
        self.exprs.push(expr);
        self.types.push(ty);
        return self.exprs.len() - 1;
    }

    // because AST is linearized reserve allows pushing a placeholder
    // while the information required for the head of a subtree is gathered
    // i.e. the index of the lhs and rhs of a binop
    fn reserve(&mut self) -> EIndex {
        return self.push(Expr::Nop);
    }

    fn next_i(&self) -> EIndex {
        return self.exprs.len();
    }

    fn expr(&mut self) -> Option<Result<EIndex>> {
        let tok = self.next_tok()?;
        let res = match tok {
            Token::Int(range) => {
                let int = Expr::Int(self.intern_int(range));
                let i = self.push_typed(int, ast::Type::UInt64);
                Ok(i)
            }
            Token::Ident(range) => {
                let str = self.intern_str(range);
                if let Some(Token::LParen) = self.peek_tok() {
                    self.fun_call(str)
                } else if let Some(Token::Eq) = self.peek_tok() {
                    self.next_tok().expect("expected Eq");
                    let assign_i = self.reserve();
                    let expr_i = self.expr()?.expect("expected expr");
                    self.exprs[assign_i] = Expr::Assign {
                        name: str,
                        value: expr_i,
                    };
                    Ok(assign_i)
                } else {
                    let ident = Expr::Ident(str);
                    let i = self.push_typed(ident, ast::Type::Unknown);
                    Ok(i)
                }
            }
            Token::True => {
                let i = self.push_typed(Expr::Bool(true), ast::Type::Bool);
                Ok(i)
            }
            Token::False => {
                let i = self.push_typed(Expr::Bool(false), ast::Type::Bool);
                Ok(i)
            }
            Token::String(range) => {
                let expr = Expr::String(self.intern_str(range));
                let i = self.push_typed(expr, ast::Type::String);
                Ok(i)
            }
            Token::Return => {
                let Ok(expr) = self.expr()? else {
                    return Some(Err(anyhow!("expected expr after return")));
                };
                let value = core::num::NonZeroUsize::new(expr);
                let i = self.push(Expr::Return { value });
                Ok(i)
            }
            Token::If => self.if_expr(),
            Token::Fun => self.fun_expr(),
            Token::Let => self.bind_expr(),
            Token::While => self.while_expr(),
            Token::LParen => Binop::try_from(self.next_tok()?)
                .and_then(|op| self.binop_expr(op))
                .with_context(|| {
                    self.tok()
                        .map(|tok| match tok {
                            Token::Ident(range) => {
                                format!("expected binop got ident {:?}", self.lxr.as_str(&range))
                            }
                            _ => format!("expected binop got {:?}", tok),
                        })
                        .unwrap_or_else(|| "expected binop got EOF".to_string())
                }),
            Token::Print => {
                eat!(self, Token::LParen, "print expects arguments");
                let Ok(expr) = self.expr()? else {
                    return Some(Err(anyhow!("expected expr after print")));
                };
                eat!(self, Token::RParen, "print expects RParen");
                let i = self.push(Expr::Print { value: expr });
                Ok(i)
            }
            _ => {
                unimplemented!("{:?} not implemented\nexprs={:?}", tok, self.exprs.iter().enumerate().collect::<Vec<_>>())
            }
        };
        return Some(res);
    }

    fn intern_int(&mut self, range: lexer::Range) -> DIndex {
        let bytes = self.lxr.slice(&range);
        let val_str = unsafe { std::str::from_utf8_unchecked(bytes) };
        let val = val_str
            // TODO: support alternative int sizes
            .parse::<u64>()
            .expect("invalid int");
        return self.data.put(&val);
    }

    fn intern_str(&mut self, range: lexer::Range) -> DIndex {
        let str = self.lxr.slice(&range);
        return self.data.put(str);
    }

    fn fun_call(&mut self, name: DIndex) -> Result<EIndex> {
        let call = self.reserve();
        let args = self.fun_call_args()?;
        self.exprs[call] = Expr::FunCall { name, args };
        return Ok(call);
    }

    fn fun_call_args(&mut self) -> Result<EIndex> {
        eat!(self, Token::LParen, "fun call args")?;
        let Some(tok) = self.peek_tok() else {
            return Err(anyhow!("expected args got EOF"));
        };

        if let Token::RParen = tok {
            self.next_tok();
            // no args
            let extra_i = self.extra.concat(&[]);
            return Ok(extra_i);
        }

        let mut args_list = Vec::with_capacity(DEFAULT_ARGS_CAP);

        loop {
            let arg = self.expr().context("expected arg")??;
            args_list.push(arg as u32);
            match self.peek_tok() {
                Some(Token::RParen) => break,
                Some(Token::Comma) => {
                    self.next_tok();
                    continue;
                }
                Some(tok) => {
                    return Err(anyhow!("unexpected token {:?}", tok));
                }
                None => {
                    return Err(anyhow!("expected RParen got EOF"));
                }
            }
        }
        eat!(self, Token::RParen, "fun call args").with_context(|| {
            return "fun call args err";
        })?;

        let extra_i = self.extra.concat(&args_list);
        return Ok(extra_i);
    }

    fn binop_expr(&mut self, op: Binop) -> Result<EIndex> {
        let expr_i = self.reserve();
        let lhs = self.expr().unwrap()?;
        let rhs = self.expr().unwrap()?;
        eat!(self, Token::RParen, "binop")?;
        self.exprs[expr_i] = Expr::Binop { op, lhs, rhs };
        Ok(expr_i)
    }

    fn while_expr(&mut self) -> Result<EIndex> {
        let while_i = self.reserve();
        let cond = self.expr().unwrap()?;
        let body = self.block()?;
        self.exprs[while_i] = Expr::While { cond, body };
        Ok(while_i)
    }

    fn if_expr(&mut self) -> Result<EIndex> {
        let if_i = self.reserve();
        let cond = self.expr().unwrap()?;

        let then_block = self.block()?;
        // FIXME: make else optional
        eat!(self, Token::Else, "if")?;
        let else_block = self.block()?;

        self.exprs[if_i] = Expr::If {
            cond,
            branch_then: then_block,
            branch_else: else_block,
        };
        Ok(if_i)
    }

    fn fun_expr(&mut self) -> Result<EIndex> {
        let fun_i = self.reserve();
        let Token::Ident(range) = eat!(self, Token::Ident(_))? else {
            unreachable!()
        };
        // dbg!(crate::utils::utf8_str!(self.lxr.slice(&range)));
        let name = self.intern_str(range);
        let args = self.fun_args()?;
        let ret_ty = self.try_parse_type_annotation()?;
        let body = self.block().context("expected function body")?;
        self.exprs[fun_i] = Expr::FunDef { name, args, body };
        self.types[fun_i] = ret_ty;
        Ok(fun_i)
    }

    fn block(&mut self) -> Result<XIndex> {
        if let Some(Token::LSquirly) = self.peek_tok() {
            /* pass */
        } else {
            let expr_i = self.expr().expect("non squirly block is expr")?;
            return Ok(self.extra.concat(&[expr_i as u32]));
        }
        eat!(self, Token::LSquirly, "block")
            .with_context(|| format!("expected {{ got {:?}", self.tok()))?;

        let mut exprs = Vec::new();
        loop {
            if let Some(Token::RSquirly) = self.peek_tok() {
                break;
            }
            let Some(expr) = self.expr() else {
                return Err(anyhow!("expected expr in block"));
            };
            exprs.push(expr? as u32);
        }
        eat!(self, Token::RSquirly, "block").with_context(|| "block err")?;
        return Ok(self.extra.concat(&exprs));
    }

    fn fun_args(&mut self) -> Result<XIndex> {
        eat!(self, Token::LParen)?;
        let Some(tok) = self.peek_tok() else {
            return Err(anyhow!("expected args got EOF"));
        };

        if let Token::RParen = tok {
            self.next_tok();
            // no args
            let extra_i = self.extra.concat(&[]);
            return Ok(extra_i);
        }

        // TODO: use array with static max args size
        // implement `ArrayList` type that is backed by constant size
        // array but has nice methods to use like vec
        let mut args_list: Vec<u32> = Vec::with_capacity(DEFAULT_ARGS_CAP);

        loop {
            let Some(Token::Ident(range)) = self.next_tok() else {
                return Err(anyhow!("expected ident in args"));
            };
            let name = self.intern_str(range);
            args_list.push(name as u32);
            let arg_type = self.try_parse_type_annotation()?;
            self.push_typed(Expr::FunArg, arg_type);
            match self.peek_tok() {
                Some(Token::RParen) => break,
                Some(Token::Comma) => {
                    self.next_tok();
                    continue;
                }
                Some(tok) => {
                    return Err(anyhow!("unexpected token {:?}", tok));
                }
                None => {
                    return Err(anyhow!("expected RParen got EOF"));
                }
            }
        }

        eat!(self, Token::RParen, "end of fun args")?;

        let extra_i = self.extra.concat(&args_list);
        return Ok(extra_i);
    }

    fn bind_expr(&mut self) -> Result<EIndex> {
        let bind_i = self.reserve();
        let Some(Token::Ident(range)) = self.next_tok() else {
            anyhow::bail!("expected ident in bind expr");
        };
        let bound_type = self.try_parse_type_annotation()?;
        let name = self.intern_str(range);

        let Some(Token::Eq) = self.next_tok() else {
            anyhow::bail!("expected `=` in bind expr, got {:?}", self.tok());
        };
        let value = self.expr().expect("no value in bind expr")?;

        self.exprs[bind_i] = Expr::Bind { name, value };
        self.types[bind_i] = bound_type;
        Ok(bind_i)
    }

    fn try_parse_type_annotation(&mut self) -> Result<ast::Type> {
        let Some(Token::Ident(range)) = self.peek_tok() else {
            return Ok(ast::Type::Unknown);
        };
        self.next_tok();
        let type_name = self.lxr.slice(&range);
        // dbg!(crate::utils::utf8_str!(type_name));
        return Ok(match type_name {
            b"int" => ast::Type::UInt64,
            b"str" => ast::Type::String,
            b"bool" => ast::Type::Bool,
            _ => anyhow::bail!("unknown type {:?}", unsafe {
                std::str::from_utf8_unchecked(type_name)
            }),
        });
    }

    #[allow(unused)]
    fn get_num_args(&self, first_arg: XIndex) -> u32 {
        return self.extra[first_arg];
    }

    fn fun_args_slice<'s>(&'s self, i: XIndex) -> &'s [u32] {
        return self.extra.get::<ast::ExtraFunArgs>(i).args;
    }

    fn fun_arg_names_iter<'s>(&'s self, i: XIndex) -> impl Iterator<Item = &'s str> {
        return self
            .fun_args_slice(i)
            .iter()
            .map(|a| self.data.get_ref::<str>(*a as usize));
    }

    fn get_ident<'s>(&'s self, i: usize) -> &'s str {
        return self.data.get_ref::<str>(i);
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Binop {
    Eq,
    Mul,
    Add,
    Sub,
    Div,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Mod,
    And,
}

impl TryFrom<Token> for Binop {
    type Error = anyhow::Error;
    fn try_from(value: Token) -> Result<Self> {
        Ok(match value {
            Token::DblEq => Binop::Eq,
            Token::Mul => Binop::Mul,
            Token::Plus => Binop::Add,
            Token::Minus => Binop::Sub,
            Token::Div => Binop::Div,
            Token::Lt => Binop::Lt,
            Token::LtEq => Binop::LtEq,
            Token::Gt => Binop::Gt,
            Token::GtEq => Binop::GtEq,
            Token::Mod => Binop::Mod,
            Token::And => Binop::And,
            _ => return Err(anyhow!("invalid binop: {:?}", value)),
        })
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

    // export
    pub(crate) use assert_matches;

    macro_rules! let_assert_matches {
        ($expr:expr, $pat:pat, $message:literal, $($arg:tt)*) => {
            let val = $expr;
            assert!(matches!(val, $pat), $message, $($arg)*);
            let $pat = val else { unreachable!("unreachable pattern") };
        };
        ($expr:expr, $pat:pat) => {
            let val = $expr;
            assert!(matches!(val, $pat), "expected {:?}, got {:?}", stringify!($pat), val);
            let $pat = val else { unreachable!("unreachable pattern") };
        };
    }

    // export
    pub(crate) use let_assert_matches;

    #[test]
    fn tok_peek() {
        let contents = "1 2 3";
        let mut parser = Parser::new(contents);
        let_assert_matches!(dbg!(parser.peek_tok()), Some(Token::Int(r1)));
        assert_eq!(dbg!(parser.next_tok()), Some(Token::Int(r1)));
        let_assert_matches!(dbg!(parser.peek_tok()), Some(Token::Int(r2)));
        assert_eq!(dbg!(parser.next_tok()), Some(Token::Int(r2)));
        let_assert_matches!(dbg!(parser.peek_tok()), Some(Token::Int(r3)));
        assert_eq!(dbg!(parser.next_tok()), Some(Token::Int(r3)));
    }

    #[test]
    fn tok_peek_2() {
        let contents = "1 2 3";
        let mut parser = Parser::new(contents);
        assert_eq!(parser.peek_tok(), parser.peek_tok());
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
        let contents = "(== 10 10)";
        let parser = parse(contents).expect("parser error");
        assert_eq!(parser.tokens.len(), 5);
        assert_eq!(
            parser.exprs[0],
            Expr::Binop {
                op: Binop::Eq,
                lhs: 1,
                rhs: 2
            }
        );
    }

    #[test]
    fn eq_with_sub_expr() {
        let contents = "(== (* 2 2) 4)";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(
            parser.exprs[0],
            Expr::Binop {
                op: Binop::Eq,
                lhs: 1,
                rhs: 4
            }
        );
        let_assert_matches!(
            parser.exprs[1],
            Expr::Binop {
                op: Binop::Mul,
                lhs: 2,
                rhs: 3
            }
        );
    }

    #[test]
    fn mul_expr() {
        let contents = "(* 10 10)";
        let parser = parse(contents).expect("parser error");
        assert_eq!(parser.tokens.len(), 5);
        let_assert_matches!(
            parser.exprs[0],
            Expr::Binop {
                op: Binop::Mul,
                lhs: 1,
                rhs: 2
            }
        );
    }

    #[test]
    fn if_expr() {
        let contents = r#"if (== 4 4) "yes" else "no""#;
        let parser = parse(contents).expect("parser error");

        let_assert_matches!(
            parser.exprs[0],
            Expr::If {
                cond,
                branch_then,
                branch_else,
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
        assert_matches!(
            parser.extra.iter_of(branch_then, &parser.exprs).next(),
            Some(Expr::String(_))
        );
        assert_matches!(
            parser.extra.iter_of(branch_else, &parser.exprs).next(),
            Some(Expr::String(_))
        );
    }

    #[test]
    fn if_expr_types() {
        let contents = r#"if (== 4 4) "yes" else "no""#;
        let parser = parse(contents).expect("parser error");

        let_assert_matches!(
            parser.exprs[0],
            Expr::If {
                cond,
                branch_then,
                branch_else,
            }
        );
        let_assert_matches!(parser.exprs[cond], Expr::Binop { op: _, lhs, rhs });
        assert_matches!(
            parser.extra.iter_of(branch_then, &parser.types).next(),
            Some(ast::Type::String)
        );
        assert_matches!(
            parser.extra.iter_of(branch_then, &parser.types).next(),
            Some(ast::Type::String)
        );

        assert_eq!(parser.types[lhs], ast::Type::UInt64);
        assert_eq!(parser.types[rhs], ast::Type::UInt64);
    }

    #[test]
    fn fun_expr() {
        let contents = "fun foo (a int, b int) int {return (+ a b)}";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunDef { name, args, body });
        let body = parser.extra.slice(body);
        let_assert_matches!(
            parser.exprs[body[0] as usize],
            Expr::Return { value: Some(value) }
        );
        assert_matches!(
            parser.exprs[usize::from(value)],
            Expr::Binop {
                op: Binop::Add,
                lhs,
                rhs
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
    fn fun_args() {
        let contents = "fun foo (a int, b int, c int, d int) int {return a}";
        let parser = parse(contents).expect("parser error");

        let ast::ExtraFunArgs { args } = parser.extra.get::<ast::ExtraFunArgs>(0);

        assert_eq!(args.len(), 4);

        let arg_strs: Vec<&str> = args
            .iter()
            .map(|a| parser.data.get_ref::<str>(*a as usize))
            .collect();
        let expected_arg_strs = vec!["a", "b", "c", "d"];

        assert_eq!(arg_strs, expected_arg_strs);
    }

    #[test]
    fn fun_call_args() {
        let contents = "foo(0, 1, 2, 3)";
        let parser = parse(contents).expect("parser error");

        let Expr::FunCall { name, args } = parser.exprs[0] else {
            unreachable!();
        };
        let args = parser.extra.slice(args);

        assert_eq!(args.len(), 4);

        for i in 0..args.len() {
            let arg = args[i as usize];
            let_assert_matches!(parser.exprs[arg as usize], Expr::Int(di));
            let data = parser.data.get::<u64>(di);
            assert_eq!(data, i as u64);
        }
    }

    #[test]
    fn fun_single_arg() {
        let contents = "fun foo (a int) int {return (+ a 1)}";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunDef { name, args, body });

        {
            let ast::ExtraFunArgs { args } = parser.extra.get::<ast::ExtraFunArgs>(args);
            assert_eq!(args.len(), 1);
        }

        let arg = parser.extra.slice(args)[0];
        assert_eq!(parser.get_ident(arg as usize), "a");
    }

    #[test]
    fn fun_name_interned() {
        let contents = "fun foo (foo int, bar int) int {return (+ foo bar)}";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunDef { name, args, body });
        assert_eq!(parser.data.get_ref::<str>(name), "foo");
    }

    #[test]
    fn fun_arg_interned() {
        let contents = "fun foo (foo int, bar int) int {return (+ foo bar)}";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunDef { name, args, body });
        let arg_names = parser.fun_arg_names_iter(args).collect::<Vec<_>>();

        assert_eq!(vec!["foo", "bar"], arg_names);
    }

    #[test]
    fn var_bind() {
        let contents = "let a int = 1";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::Bind { name, value });
        assert_eq!(parser.data.get_ref::<str>(name), "a");
        let_assert_matches!(parser.exprs[value], Expr::Int(int));
        assert_eq!(parser.data.get::<u64>(int), 1);
    }

    #[test]
    fn var_bind_types() {
        let contents = "let a int = 1";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::Bind { name, value });
        assert_eq!(parser.types[value], ast::Type::UInt64);
    }

    #[test]
    fn fun_call() {
        let contents = "foo(1, 2)";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunCall { name, args });
        assert_eq!(parser.data.get_ref::<str>(name), "foo");
        // NOTE: asserts args come after the arg exprs in the exprs vec
        assert_eq!(parser.fun_args_slice(args), &[1, 2]);
    }

    #[test]
    fn fun_call_arg_expr() {
        let contents = "foo((+ 1 1), 2)";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunCall { name, args });
        assert_eq!(parser.data.get_ref::<str>(name), "foo");
        let Some(Expr::Binop { .. }) = parser.extra.iter_of(args, &parser.exprs).next() else {
            unreachable!();
        };
    }

    #[test]
    fn fun_call_arg_expr_types() {
        let contents = "foo((+ 1 1), 2)";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunCall { name, args });
        let &[binop_arg, const_arg] = parser.fun_args_slice(args) else {
            unreachable!();
        };
        let_assert_matches!(
            parser.exprs[binop_arg as usize],
            Expr::Binop { op: _, lhs, rhs }
        );
        assert_eq!(parser.types[const_arg as usize], ast::Type::UInt64);
        assert_eq!(parser.types[lhs], ast::Type::UInt64);
        assert_eq!(parser.types[rhs], ast::Type::UInt64);
    }

    #[test]
    fn fun_call_no_args() {
        let contents = "foo()";
        let parser = parse(contents).expect("parse error");
        let_assert_matches!(parser.exprs[0], Expr::FunCall { name, args });
        assert_eq!(parser.data.get_ref::<str>(name), "foo");
        assert_eq!(parser.fun_args_slice(args), &[]);
    }

    #[test]
    fn fib_if() {
        let contents = r#"
            if (<= n 1) {
                return n
            } else {
                return (+
                  fib(( - n 1 ))
                  fib(( - n 2 ))
                )
            }
        "#;
        let parser = parse(contents).expect("parser error");
    }

    #[test]
    fn echo() {
        let contents = r#"fun echo(a int) int {return a}"#;
        let parser = parse(contents).expect("parser error");
        assert_matches!(
            parser.exprs[0],
            Expr::FunDef {
                name: _,
                args: _,
                body: _
            }
        );
        assert_matches!(parser.exprs[1], Expr::FunArg);
        assert_matches!(parser.exprs[2], Expr::Ident(_));
    }

    #[test]
    fn fun_type_annotations() {
        let contents = "fun foo (a int, b int) int {return (+ a b)}";
        let parser = parse(contents).expect("parser error");

        assert_matches!(
            parser.exprs[0],
            Expr::FunDef {
                name: _,
                args: _,
                body: _
            }
        );
        assert_eq!(parser.types[0], ast::Type::UInt64);

        assert_eq!(parser.exprs[1], Expr::FunArg);
        assert_eq!(parser.types[1], ast::Type::UInt64);

        assert_eq!(parser.exprs[2], Expr::FunArg);
        assert_eq!(parser.types[2], ast::Type::UInt64);
    }

    #[test]
    fn bind_type_annotations() {
        let contents = "let a int = 1";
        let parser = parse(contents).expect("parser error");
        assert_matches!(parser.exprs[0], Expr::Bind { name: _, value: _ });
        assert_eq!(parser.types[0], ast::Type::UInt64);
    }

    #[test]
    fn multiple_fundef() {
        let contents = r#"fun foo (a int) int {return a} fun bar () {foo(1)}"#;
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(
            parser.exprs[0],
            Expr::FunDef {
                name: _,
                args: _,
                body
            }
        );
        // {
        //     let ast = parser.clone().ast();
        //     ast.print();
        // };
        let bar_id = parser.data.get_ident_id("bar").unwrap();
        let Some(&Expr::FunDef { body: bar_body, .. }) = parser
            .exprs
            .iter()
            .find(|e| matches!(e, Expr::FunDef {name, ..} if parser.get_ident(*name) == "bar"))
        else {
            unreachable!("no bar fun");
        };
        // let bar_first_stmt = ;
        let_assert_matches!(
            parser.extra.iter_of(bar_body, &parser.exprs).next(),
            Some(&Expr::FunCall { name, args })
        );
        assert_eq!(parser.data.get_ref::<str>(name), "foo");
        assert_matches!(
            parser.extra.iter_of(args, &parser.exprs).next(),
            Some(Expr::Int(_))
        );
    }

    #[test]
    fn if_expr_gib() {
        let contents = r#"
            let foo int = if true {
                gib 1
            } else {
                gib 2
            }
        "#;
        let parser = parse(contents).expect("parser error");
    }

    #[test]
    fn pipe() {
        let contents = r#"let a int = 1 |> foo() |> bar()"#;
        let parser = parse(contents).expect("parser error");
    }
}
