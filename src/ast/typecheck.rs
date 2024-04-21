use anyhow::Result;

use crate::ast;
use super::{DIndex, EIndex, Ast, Type};
use super::Expr::*;

pub fn typecheck(ast: &mut Ast) -> Result<()> {
    let mut i = 0;
    while i < ast.exprs.len() {
        let expr_i = i;
        let expr = ast.exprs[expr_i];
        // TODO: switch to check for statements,
        // call check_expr in else branch
        match expr {
            Nop => anyhow::bail!("nop has no type"),
            Int(_) | String(_) => anyhow::bail!("top level literal not valid. malformed AST"),
            Binop {op, lhs, rhs} => {
                check_binop(ast, &mut i, expr_i, op, lhs, rhs)?;
            },
            Bind => todo!("bind"),
            Ident(_) => todo!("name resolution"),
            If {..} => todo!("if"),
            FunCall { .. } => todo!("fun call"),
            FunDef {..} => todo!("fun def"),
        }
        i+=1;
    }
    Ok(())
}

fn check_expr(ast: &mut Ast, cursor: &mut usize, expr_i: EIndex) -> Result<Type> {
    let expr = ast.exprs[expr_i];
    match expr {
        Nop => anyhow::bail!("nop has no type"),
        Int(_) | String(_) => {
            mark_visited(cursor, expr_i);
            Ok(ast.types[expr_i])
        },
        Binop {op, lhs, rhs} => check_binop(ast, cursor, expr_i, op, lhs, rhs),
        Bind => todo!("bind"),
        Ident(_) => todo!("name resolution"),
        If {..} => todo!("if"),
        FunCall { .. } => todo!("fun call"),
        FunDef {..} => todo!("fun def"),
    }
}

fn check_binop(ast: &mut Ast, cursor: &mut usize, binop_i: EIndex, op: ast::Binop, lhs_i: EIndex, rhs_i: EIndex) -> Result<Type> {
    let lhs_type = check_expr(ast, cursor, lhs_i)?;
    let rhs_type = check_expr(ast, cursor, rhs_i)?;
    if lhs_type != rhs_type {
        // TODO: better error message
        anyhow::bail!("mismatched types in binop");
    }
    let binop_type = lhs_type;
    ast.types[binop_i] = binop_type;
    return Ok(binop_type);
}

fn mark_visited(cursor: &mut usize, i: usize) {
    *cursor = usize::max(*cursor, i);
}


#[derive(Debug)]
struct ScopeStack {
    stack_map: Vec<DIndex>,
    starts: Vec<usize>,
    cur: usize,
}

impl ScopeStack {
    const NOT_BOUND: DIndex = usize::MAX;

    fn new() -> Self {
        Self {
            stack_map: vec![],
            starts: vec![0],
            cur: 0,
        }
    }

    fn skip<const N: usize>(&mut self) {
        let vals = [Self::NOT_BOUND; N];
        self.stack_map.extend(vals);
    }

    /// like start_new but does not set `cur`
    fn start_subscope(&mut self) {
        self.starts.push(self.stack_map.len());
    }

    fn end_subscope(&mut self) {
        let start = self.starts.pop().expect("no starts");
        self.stack_map.truncate(start);
    }

    fn start_new(&mut self) {
        self.starts.push(self.cur);
        self.cur = self.stack_map.len();
    }

    fn end(&mut self) {
        let start = self.cur;
        self.cur = self.starts.pop().expect("no starts");
        self.stack_map.truncate(start);
    }

    /// NOTE: does not check that name is not already in scope,
    /// therefore allowing shadowing
    fn bind_local(&mut self, name: DIndex) -> u32 {
        let i = self.stack_map.len();
        self.stack_map.push(name);
        return (i - self.cur) as u32;
    }

    fn get(&self, name: DIndex) -> Option<u32> {
        debug_assert_ne!(name, Self::NOT_BOUND);
        let pos_r = self.stack_map.iter().rev().position(|&n| n == name);
        let Some(pos_r) = pos_r else {
            return None;
        };
        let pos = self.stack_map.len() - pos_r - 1;
        if pos < self.cur {
            unimplemented!("tried to get reference to variable outside of stack frame. globals not implemented");
        }
        return Some((pos - self.cur) as u32);
    }

    fn set(&mut self, i: u32, name: DIndex) {
        self.stack_map[i as usize] = name;
    }

    fn is_last(&self, i: u32) -> bool {
        let i = self.cur + i as usize;
        let len = self.stack_map.len();
        return i + 1 == len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse<'a>(contents: &'a str) -> Result<Ast> {
        let mut parser = crate::parser::Parser::new(contents);
        parser.parse()
    }

    // TODO: make macros for checking that the typecheck fails with specific errors
    macro_rules! assert_accepts {
        ($src:expr) => {
            let mut ast = parse($src).expect("parser error");
            let res = super::typecheck(&mut ast);
            assert!(res.is_ok(), "{}", res.unwrap_err());
        };
    }

    macro_rules! assert_rejects {
        ($src:expr) => {
            let mut ast = parse($src).expect("parser error");
            let res = super::typecheck(&mut ast);
            assert!(res.is_err());

        };
    }

    

    #[test]
    fn add() {
        assert_accepts!("(+ 1 2)");
    }

    #[test]
    fn add_int_str() {
        assert_rejects!(r#"(+ 1 "hello")"#);
        assert_rejects!(r#"(+ "hello" 1)"#);
    }

    #[test]
    fn eq() {
        assert_accepts!("(== 1 2)");
    }

    #[test]
    fn type_inferred_from_op() {
        assert_rejects!("(+ (== 1 2) 2)");
    }

    #[test]
    fn fun_return_type_misused() {
        assert_rejects!(r#"(fun foo () 1) (== (foo) "some_str")"#);
    }

    #[test]
    fn multiple_fun_return_type_misused() {
        assert_rejects!(r#"(fun foo () 1) (fun bar () "str") (== (foo) (bar))"#);
    }

    #[test]
    fn mismatched_return_types() {
        assert_rejects!(r#"(fun foo() (if (== 1 2) 1 "str"))"#);
    }

}
