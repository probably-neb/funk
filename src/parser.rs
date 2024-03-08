use anyhow::{anyhow, Context, Result};

use crate::{
    ast::{Ast, DIndex, DataPool, self},
    lexer::{self, Lexer, Token},
};

/// Expression index into `exprs`
pub type EIndex = usize;

/// Token index into `tokens`
pub type TIndex = usize;

/// index into `extra`
pub type XIndex = usize;

pub struct Parser<'a> {
    lxr: Lexer<'a>,
    tok_i: TIndex,
    tokens: Vec<Token>,
    exprs: Vec<Expr>,
    data: DataPool,
    extra: ast::Extra
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
        op: Binop,
        lhs: EIndex,
        rhs: EIndex,
    },
    FunDef {
        name: DIndex,
        args: XIndex,
        body: EIndex,
    },
    FunCall {
        name: DIndex,
        args: XIndex,
    },
    Bind {
        name: DIndex,
        value: EIndex,
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
    ($self:ident, $tok:pat, $msg:literal) => {
        match $self.tok() {
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
            extra: ast::Extra::new()
        }
    }
    pub fn new(contents: &'a str) -> Parser<'a> {
        Parser::_new(Lexer::new(contents))
    }

    pub fn parse(mut self) -> Result<crate::ast::Ast> {
        self._parse()?;
        Ok(Ast::new(self.exprs, self.data, self.extra))
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

    fn tok(&mut self) -> Option<Token> {
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
            return None;
        }
        self.tokens.push(tok);
        return Some(tok);
    }

    fn peek_tok(&mut self) -> Option<Token> {
        let (peek_i, has_peeked) = self._peek_i();
        if has_peeked {
            return Some(self.tokens[peek_i])
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
        let in_block = tok == Token::LParen ;
        if in_block {
            tok = self.tok()?;
        }
        let expr = match tok {
            Token::Int(range) => Ok(Expr::Int(self.intern_int(range))),
            Token::Ident(range) if in_block => {
                let name = self.intern_str(range);
                return Some(self.fun_call(name));
            }
            Token::Ident(range) => Ok(Expr::Ident(self.intern_str(range))),
            Token::String(range) => Ok(Expr::String(self.intern_str(range))),
            Token::If => return Some(self.if_expr()),
            Token::Fun => return Some(self.fun_expr()),
            Token::Let => return Some(self.bind_expr()),
            Token::Eq
            | Token::DblEq
            | Token::Mul
            | Token::Plus
            | Token::Minus
            | Token::Div
            | Token::LtEq
            | Token::GtEq
            | Token::Lt
            | Token::Gt => return Some(self.binop_expr(Binop::from(tok))),
            _ => {
                unimplemented!("{:?} not implemented", tok)
            }
        };

        match expr {
            Err(err) => return Some(Err(err)),
            Ok(expr) => return Some(Ok(self.push(expr))),
        }
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
        self.exprs[call] = Expr::FunCall {
            name,
            args
        };
        eat!(self, Token::RParen, "end of fun call").with_context(|| {
            print_tree(self);
            "fun call err"
        })?;
        return Ok(call);
    }

    fn fun_call_args(&mut self) -> Result<EIndex> {
        let Some(tok) = self.peek_tok() else {
            return Err(anyhow!("expected args got EOF"));
        };

        let extra_args = self.extra.reserve();

        if let Token::RParen = tok {
            // no args
            self.extra[extra_args] = 0;
            return Ok(extra_args);
        }

        let mut num = 0;

        while !matches!(self.peek_tok(), Some(Token::RParen)) {
            let arg = self.expr().context("expected arg")??;
            self.extra.append(arg as u32);
            num += 1;
        }

        self.extra[extra_args] = num;

        Ok(extra_args)
    }

    fn binop_expr(&mut self, op: Binop) -> Result<EIndex> {
        let expr_i = self.reserve();
        let lhs = self.expr().unwrap()?;
        let rhs = self.expr().unwrap()?;
        eat!(self, Token::RParen, "binop")?;
        self.exprs[expr_i] = Expr::Binop { op, lhs, rhs };
        Ok(expr_i)
    }

    fn if_expr(&mut self) -> Result<EIndex> {
        let if_i = self.reserve();
        let cond = self.expr().unwrap()?;
        let branch_true = self.expr().unwrap()?;
        let branch_false = self.expr().unwrap()?;
        eat!(self, Token::RParen, "if")?;
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
        let body = self.expr().context("expected function body")??;
        eat!(self, Token::RParen, "fun").with_context(|| {
            print_tree(self);
            return "fun err";
        })?;
        self.exprs[fun_i] = Expr::FunDef { name, args, body };
        Ok(fun_i)
    }

    fn fun_args(&mut self) -> Result<XIndex> {
        eat!(self, Token::LParen)?;
        let Some(tok) = self.tok() else {
            return Err(anyhow!("expected args got EOF"));
        };

        let extra_args_len = self.extra.reserve();

        if let Token::RParen = tok {
            // no args
            self.extra[extra_args_len] = 0;
            return Ok(extra_args_len);
        }
        let Token::Ident(first_range) = tok else {
            return Err(anyhow!("incomplete args"));
        };
        let first_name = self.intern_str(first_range);
        self.extra.append(first_name as u32);

        let mut num = 1;

        while let Some(Token::Ident(range)) = self.tok() {
            num += 1;
            let name = self.intern_str(range);
            self.extra.append(name as u32);
        }

        self.extra[extra_args_len] = num;
        debug_assert_eq!(self.tokens[self.tok_i], Token::RParen);
        Ok(extra_args_len)
    }


    fn bind_expr(&mut self) -> Result<EIndex> {
        let bind_i = self.reserve();
        let Token::Ident(range) = eat!(self, Token::Ident(_))? else {
            unreachable!()
        };
        let name = self.intern_str(range);
        let value = self.expr().expect("no value in bind expr")?;
        eat!(self, Token::RParen)?;
        self.exprs[bind_i] = Expr::Bind { name, value };
        Ok(bind_i)
    }

    #[allow(unused)]
    fn get_num_args(&self, first_arg: XIndex) -> u32 {
        return self.extra[first_arg];
    }

    fn fun_args_slice<'s>(&'s self, i: XIndex) -> &'s [u32] {
        return self.extra.get::<ast::ExtraFunArgs>(i).args;
    }

    fn fun_arg_names_iter<'s>(&'s self, i: XIndex) -> impl Iterator<Item = &'s str> {
        return self.fun_args_slice(i).iter().map(|a| self.data.get_ref::<str>(*a as usize));
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
}

impl From<Token> for Binop {
    fn from(value: Token) -> Self {
        match value {
            Token::DblEq => Binop::Eq,
            Token::Mul => Binop::Mul,
            Token::Plus => Binop::Add,
            Token::Minus => Binop::Sub,
            Token::Div => Binop::Div,
            Token::Lt => Binop::Lt,
            Token::LtEq => Binop::LtEq,
            Token::Gt => Binop::Gt,
            Token::GtEq => Binop::GtEq,
            _ => unreachable!("invalid binop: {:?}", value),
        }
    }
}

use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
struct TreeNode<T: std::fmt::Display> {
    data: T,
    children: Vec<Rc<RefCell<TreeNode<T>>>>,
}

impl<T: std::fmt::Display> TreeNode<T> {
    // Create a new tree node
    fn new(data: T) -> Self {
        TreeNode {
            data,
            children: vec![],
        }
    }

    // Add a child node directly to this node
    fn add_node(&mut self, data: TreeNode<T>) {
        self.children.push(Rc::new(RefCell::new(data)));
    }

    // Print the tree
    fn print(&self, prefix: String, is_last: bool) {
        println!(
            "{}{}{}",
            prefix,
            if is_last { "└─ " } else { "├─ " },
            self.data
        );
        let new_prefix = if is_last { "    " } else { "|   " };

        let children = &self.children;
        let last_index = children.len().saturating_sub(1);

        for (index, child) in children.iter().enumerate() {
            TreeNode::print(
                &child.borrow(),
                format!("{}{}", prefix, new_prefix),
                index == last_index,
            );
        }
    }
}

fn print_tree(parser: &Parser<'_>) {
    let tree = into_tree(parser);
    tree.print("".to_string(), false);
}

fn into_tree(parser: &Parser<'_>) -> TreeNode<String> {
    let mut visited = vec![false; parser.exprs.len()];
    let mut root = TreeNode::new("Root".to_string());
    for (i, expr) in parser.exprs.iter().enumerate() {
        if !visited[i] {
            let node = expr_into_treenode(expr.clone(), parser, &mut visited);
            root.add_node(node);
        }
    }
    return root;
}

fn expr_into_treenode(expr: Expr, parser: &Parser<'_>, visited: &mut [bool]) -> TreeNode<String> {
    let data = repr_expr(expr, parser);
    let mut node = TreeNode::new(data);
    macro_rules! visit {
        ($node:ident, $i:expr) => {
            visited[$i] = true;
            let child = expr_into_treenode(parser.exprs[$i], parser, visited);
            $node.add_node(child);
        };
        ($i:expr) => {
            visit!(node, $i);
        };
    }
    use Expr::*;
    match expr {
        Nop | Int(_) | String(_) | Ident(_) => {}
        FunCall { args, .. } => {
            let names: Vec<&str> = parser.fun_arg_names_iter(args).collect();
            node.add_node(TreeNode::new(format!("args: [{}]", names.join(", "))));
        }
        If {
            cond,
            branch_true,
            branch_false,
        } => {
            visit!(cond);
            visit!(branch_true);
            visit!(branch_false);
        }
        Binop { lhs, rhs, ..} => {
            visit!(lhs);
            visit!(rhs);
        }
        FunDef { args, body, ..} => {
            let mut arg_node = TreeNode::new("Args".to_string());
            for arg_i in parser.fun_args_slice(args) {
                let arg = expr_into_treenode(expr, parser, visited);
                arg_node.add_node(arg);
                visited[*arg_i as usize] = true;
            }
            node.add_node(arg_node);

            visit!(body);
        }
        Bind { name, value } => {
            let name = expr_into_treenode(parser.exprs[name], parser, visited);
            let value = expr_into_treenode(parser.exprs[value], parser, visited);
            node.add_node(name);
            node.add_node(value);
        }
    }
    return node;
}

fn repr_expr(expr: Expr, parser: &Parser<'_>) -> String {
    match expr {
        Expr::Nop => "Nop".to_string(),
        Expr::Int(i) => format!("Int {}", parser.data.get::<u64>(i)),
        Expr::Binop { op,..} => format!("{:?}", op),
        Expr::If {
            ..
        } => "If".to_string(),
        Expr::FunDef { name, ..} => format!("Fun {:?}", parser.get_ident(name)),
        Expr::FunCall { name, .. } => format!("Call {:?}", parser.get_ident(name)),
        Expr::Ident(i) => format!("Ident {:?}", parser.get_ident(i)),
        Expr::String(i) => format!("Str \"{:?}\"", parser.get_ident(i)),
        Expr::Bind { name, .. } => format!("let {:?}", parser.get_ident(name)),
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
        assert_eq!(dbg!(parser.tok()), Some(Token::Int(r1)));
        let_assert_matches!(dbg!(parser.peek_tok()), Some(Token::Int(r2)));
        assert_eq!(dbg!(parser.tok()), Some(Token::Int(r2)));
        let_assert_matches!(dbg!(parser.peek_tok()), Some(Token::Int(r3)));
        assert_eq!(dbg!(parser.tok()), Some(Token::Int(r3)));
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
        let contents = r#"(if (== 4 4) "yes" "no")"#;
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
    fn fun_args() {
        let contents = "(fun foo (a b c d) (a))";
        let parser = parse(contents).expect("parser error");

        let ast::ExtraFunArgs {args} = parser.extra.get::<ast::ExtraFunArgs>(0);

        assert_eq!(args.len(), 4);

        let arg_strs: Vec<&str> = args.iter().map(|a| parser.data.get_ref::<str>(*a as usize)).collect();
        let expected_arg_strs = vec!["a", "b", "c", "d"];

        assert_eq!(arg_strs, expected_arg_strs);
    }

    #[test]
    fn fun_call_args() {
        let contents = "(foo 0 1 2 3)";
        let parser = parse(contents).expect("parser error");

        let ast::ExtraFunArgs {args} = parser.extra.get::<ast::ExtraFunArgs>(0);

        assert_eq!(args.len(), 4);

        dbg!(args);
        for i in 0..args.len() {
            let arg = args[i as usize];
            let_assert_matches!(parser.exprs[arg as usize], Expr::Int(di));
            dbg!(di);
            let data = dbg!(parser.data.get::<u64>(di));
            assert_eq!(data, i as u64);
        }
    }

    #[test]
    fn fun_single_arg() {
        let contents = "(fun foo (a) (+ a 1))";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunDef { name, args, body });

        {
            let ast::ExtraFunArgs {args} = parser.extra.get::<ast::ExtraFunArgs>(args);
            assert_eq!(args.len(), 1);
        }

        let arg = parser.fun_args_slice(args)[0];
        assert_eq!(parser.get_ident(arg as usize), "a");
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
        let arg_names = parser.fun_arg_names_iter(args).collect::<Vec<_>>();

        assert_eq!(vec!["foo", "bar"], arg_names);
    }

    #[test]
    fn var_bind() {
        let contents = "(let a 1)";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::Bind { name, value });
        assert_eq!(parser.data.get_ref::<str>(name), "a");
        let_assert_matches!(parser.exprs[value], Expr::Int(int));
        assert_eq!(parser.data.get::<u64>(int), 1);
    }

    #[test]
    fn fun_call() {
        let contents = "(foo 1 2)";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunCall { name, args });
        assert_eq!(parser.data.get_ref::<str>(name), "foo");
        // NOTE: asserts args come after the arg exprs in the exprs vec
        assert_eq!(parser.fun_args_slice(args), &[1, 2]);
    }

    #[test]
    fn fun_call_arg_expr() {
        let contents = "(foo (+ 1 1) 2)";
        let parser = parse(contents).expect("parser error");
        let_assert_matches!(parser.exprs[0], Expr::FunCall { name, args });
        assert_eq!(parser.data.get_ref::<str>(name), "foo");
        let &[binop_arg, _] = parser.fun_args_slice(args) else {
            unreachable!();
        };
        assert_matches!(parser.exprs[binop_arg as usize], Expr::Binop { op: _, lhs: _, rhs: _ });
    }

    #[test]
    fn fun_call_no_args() {
        let contents = "(foo)";
        let parser = parse(contents).expect("parse error");
        let_assert_matches!(parser.exprs[0], Expr::FunCall { name, args });
        assert_eq!(parser.data.get_ref::<str>(name), "foo");
        assert_eq!(parser.fun_args_slice(args), &[]);
    }

    #[test]
    fn fib_if() {
        let contents = r#" ( if (<= n 1)
                    (return n)
                    (return (+
                              (fib (- n 1))
                              (fib (- n 2))
                              ))
                )
        "#;
        let parser = parse(contents).expect("parser error");
    }

}
