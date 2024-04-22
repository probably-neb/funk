use anyhow::{Result, Context};

use super::{Ast, DIndex, EIndex, Expr, Expr::*, RefIdent, Type, Extra};
use crate::ast;

pub fn typecheck(ast: &mut Ast) -> Result<()> {
    let mut i = 0;
    let mut mod_scopes = ModuleScopes {
        globals: ScopeStack::new(),
        functions: ScopeStack::new(),
    };
    let mut local_scopes = ScopeStack::new();

    while i < ast.exprs.len() {
        let expr_i = i;
        let expr = ast.exprs[expr_i];

        let types = &mut ast.types;
        let exprs = &ast.exprs;
        let refs = &mut ast.refs;
        let extra = &ast.extra;

        // TODO: switch to check for statements,
        // call check_expr in else branch
        match expr {
            Nop => anyhow::bail!("nop has no type"),
            Int(_) | String(_) | Ident(_) => anyhow::bail!("top level literal not valid. malformed AST"),
            Binop { op, lhs, rhs } => {
                check_binop(&exprs, types, refs, &mut i, expr_i, op, lhs, rhs)?;
            }
            FunDef { name, args, body } => {
                i+=1;
                for &arg in extra.fun_args_slice(args) {
                    assert!(exprs[i] == FunArg, "expected fun arg");
                    local_scopes.bind(arg as usize, i);
                    i+=1;
                }
                assert_eq!(i, body);
                check_block(exprs, types, refs, extra, &mut i, &mut mod_scopes, &mut local_scopes)?;
                local_scopes.clear();
            },
            If { .. } => todo!("if"),
            Bind { .. } => todo!("bind"),
            FunCall { .. } => todo!("fun call"),
            FunArg => anyhow::bail!("expected top level node found fun arg"),
            BlockEnd => unreachable!("block end"),
            FunEnd => unreachable!("fun end"),
        }
        i += 1;
    }
    Ok(())
}

fn check_block(
    exprs: &[Expr],
    types: &mut [Type],
    refs: &mut [RefIdent],
    extra: &Extra,
    cursor: &mut usize,
    module_scopes: &mut ModuleScopes,
    local_scopes: &mut ScopeStack,
) -> Result<()> {
    while exprs[*cursor] != BlockEnd && exprs[*cursor] != FunEnd {
        let expr_i = *cursor;
        let expr = exprs[expr_i];

        // TODO: switch to check for statements,
        // call check_expr in else branch
        match expr {
            Expr::Nop => anyhow::bail!("nop has no type"),
            Expr::Int(_) | Expr::String(_) => {
            },
            Expr::Ident(name) => {
                // TODO: implicit returns?
                local_scopes.get(name).context("unbound identifier")?;
            },
            Expr::Binop { op, lhs, rhs } => {
                check_binop(&exprs, types, refs, cursor, expr_i, op, lhs, rhs)?;
            }
            Expr::If { .. } => todo!("if"),
            Expr::Bind { .. } => todo!("bind"),
            Expr::FunCall { .. } => todo!("fun call"),
            Expr::FunArg => anyhow::bail!("expected block level node found fun arg"),
            Expr::FunDef {..} => anyhow::bail!("expected block level node found fun def"),
            Expr::BlockEnd => unreachable!("block end"),
            Expr::FunEnd => unreachable!("fun end"),
        }
        *cursor += 1;
        if *cursor == exprs.len() {
            panic!("block end not found");
        }
    }
    *cursor += 1;
    Ok(())
}

fn check_expr(
    exprs: &[Expr],
    types: &mut [Type],
    refs: &mut [RefIdent],
    cursor: &mut usize,
    expr_i: EIndex,
) -> Result<Type> {
    let expr = exprs[expr_i];
    match expr {
        Expr::Nop => anyhow::bail!("nop has no type"),
        Expr::Int(_) | Expr::String(_) => {
            mark_visited(cursor, expr_i);
            Ok(types[expr_i])
        },
        Expr::Binop { op, lhs, rhs } => check_binop(exprs, types, refs, cursor, expr_i, op, lhs, rhs),
        Expr::Bind {..} => todo!("bind"),
        Expr::Ident(_) => todo!("name resolution"),
        Expr::If { .. } => todo!("if"),
        Expr::FunCall { .. } => todo!("fun call"),
        Expr::FunDef { .. } => todo!("fun def"),
        Expr::FunArg => anyhow::bail!("expected expr found fun arg"),
        Expr::BlockEnd => unreachable!("block end"),
        Expr::FunEnd => unreachable!("fun end"),
    }
}

fn check_binop(
    exprs: &[Expr],
    types: &mut [Type],
    refs: &mut [RefIdent],
    cursor: &mut usize,
    binop_i: EIndex,
    op: ast::Binop,
    lhs_i: EIndex,
    rhs_i: EIndex,
) -> Result<Type> {
    let lhs_type = check_expr(exprs, types, refs, cursor, lhs_i)?;
    let rhs_type = check_expr(exprs, types, refs, cursor, rhs_i)?;
    if lhs_type != rhs_type {
        // TODO: better error message
        anyhow::bail!("mismatched types in binop");
    }
    let binop_type = match op {
        ast::Binop::Add | ast::Binop::Sub | ast::Binop::Mul | ast::Binop::Div => match lhs_type {
            Type::UInt64 => Type::UInt64,
            _ => anyhow::bail!("invalid type for arithmetic op"),
        },
        ast::Binop::Eq => Type::Bool,
        ast::Binop::Lt | ast::Binop::Gt | ast::Binop::GtEq | ast::Binop::LtEq => match lhs_type {
            Type::UInt64 => Type::Bool,
            _ => anyhow::bail!("invalid type for comparison op"),
        },
    };
    types[binop_i] = binop_type;
    return Ok(binop_type);
}

fn mark_visited(cursor: &mut usize, i: usize) {
    *cursor = usize::max(*cursor, i);
}

/// Collection of scopes related to a module
#[derive(Debug)]
struct ModuleScopes {
    globals: ScopeStack,
    /// also globals, but not variables
    functions: ScopeStack,
}

#[derive(Debug)]
struct ScopeStack {
    indices: Vec<EIndex>,
    names: Vec<DIndex>,

    starts: Vec<usize>,
    cur: usize,
}

impl ScopeStack {
    const NOT_BOUND: DIndex = usize::MAX;

    fn new() -> Self {
        Self {
            indices: vec![],
            names: vec![],
            starts: vec![0],
            cur: 0,
        }
    }

    /// like start_new but does not set `cur`
    fn start_subscope(&mut self) {
        self.starts.push(self.names.len());
    }

    fn end_subscope(&mut self) {
        let prev_start = self.starts.pop().expect("no starts");
        self.names.truncate(prev_start);
        self.indices.truncate(prev_start);
    }

    fn start_new(&mut self) {
        self.starts.push(self.cur);
        self.cur = self.names.len();
    }

    fn end(&mut self) {
        let start = self.cur;
        self.cur = self.starts.pop().expect("no starts");
        self.names.truncate(start);
        self.indices.truncate(start);
    }

    /// NOTE: does not check that name is not already in scope,
    /// therefore allowing shadowing
    fn bind(&mut self, name: DIndex, expr_i: EIndex) {
        let i = self.indices.len();
        self.names.push(name);
        self.indices.push(expr_i);
    }

    fn get(&self, name: DIndex) -> Option<EIndex> {
        debug_assert_ne!(name, Self::NOT_BOUND);
        let pos_r = self.names.iter().rev().position(|&n| n == name);
        let Some(pos_r) = pos_r else {
            return None;
        };
        let pos = self.names.len() - pos_r - 1;
        if pos < self.cur {
            unimplemented!("tried to get reference to variable outside of stack frame. globals not implemented");
        }
        return Some(self.indices[pos]);
    }

    fn clear(&mut self) {
        self.names.clear();
        self.indices.clear();
        // keep the first start (0)
        self.starts.truncate(1);
        debug_assert_eq!(self.starts[0], 0);
        self.cur = 0;
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn parse<'a>(contents: &'a str) -> Result<Ast> {
        let mut parser = crate::parser::Parser::new(contents);
        parser.parse()
    }

    // TODO: make macros for checking that the typecheck fails with specific errors
    macro_rules! assert_accepts {
        ($src:expr) => {
            {
                let mut ast = parse($src).expect("parser error");
                ast.print();
                let res = super::typecheck(&mut ast);
                assert!(res.is_ok(), "{}", res.unwrap_err());
                ast
            }
        };
    }

    macro_rules! assert_rejects {
        ($src:expr) => {
            {
                let mut ast = parse($src).expect("parser error");
                ast.print();
                let res = super::typecheck(&mut ast);
                assert!(res.is_err());
                ast
            }
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
        // FIXME: add optional return type annotations
        // instead of inferring function return types for now
        assert_rejects!(r#"(fun foo () 1) (== (foo) "some_str")"#);
    }

    #[test]
    fn multiple_fun_return_type_misused() {
        // FIXME: this test is passing but idk why (probably because it sees unknown as invalid
        // type)
        // but this isn't right! Should make `assert_rejects_with` as well as typecheck error union
        // type and check that error matches pattern
        assert_rejects!(r#"(fun foo () 1) (fun bar () "str") (== (foo) (bar))"#);
    }

    #[test]
    fn mismatched_return_types() {
        assert_rejects!(r#"(fun foo() (if (== 1 2) 1 "str"))"#);
    }

    #[test]
    fn fun_arg_is_return_type() {
        assert_accepts!(r#"(fun foo (a) a)"#);
    }

    #[test]
    fn unbound_ident() {
        assert_rejects!("(fun foo () a)");
    }
}
