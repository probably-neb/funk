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
            Expr::Nop => anyhow::bail!("nop has no type"),
            Expr::Int(_) | Expr::String(_) | Expr::Ident(_) => anyhow::bail!("top level literal not valid. malformed AST"),
            Expr::Binop { op, lhs, rhs } => {
                check_binop(&exprs, types, refs, &mut i, expr_i, op, lhs, rhs)?;
            }
            Expr::FunDef { name, args, body } => {
                let fun_i = i;
                mod_scopes.functions.bind(name, fun_i);

                i+=1;
                for &arg in extra.fun_args_slice(args) {
                    assert!(exprs[i] == FunArg, "expected fun arg");
                    local_scopes.bind(arg as usize, i);
                    i+=1;
                    types[i] = Type::Unknown;
                }
                assert_eq!(i, body.start_i());
                let res_type = check_block(exprs, types, refs, extra, &mut i, &mut mod_scopes, &mut local_scopes, body.end_i())?;
                local_scopes.clear();
                let ret_type = &mut types[fun_i];
                if *ret_type == Type::Unknown {
                    *ret_type = res_type;
                } else if res_type != types[fun_i] {
                    anyhow::bail!("function return type does not match actual return type");
                }
            },
            Expr::If { .. } => todo!("if"),
            Expr::Bind { .. } => todo!("bind"),
            Expr::FunCall { name, args } => {
                let fun_i = mod_scopes.functions.get(name).with_context(|| "ruh roh raggiy")?;
                let Expr::FunDef { args: params_i, ..} = exprs[fun_i] else {
                    anyhow::bail!("expected fun def");
                };
                let num_params = {
                    let num = extra.fun_num_args(params_i);
                    num as usize
                };
                dbg!(num_params);
                let param_types_range = fun_i + 1..=(fun_i + num_params);
                for (&arg, param_type_i) in extra.fun_args_slice(args).iter().zip(param_types_range) {
                    let arg_type = check_expr(exprs, types, refs, &mut i, arg as usize)?;
                    let param_type = types[param_type_i];
                    dbg!(arg_type, param_type);
                    if arg_type != param_type {
                        anyhow::bail!("mismatched types in function call");
                    }
                }
            },
            Expr::FunArg => anyhow::bail!("expected top level node found fun arg"),
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
    until: usize,
) -> Result<Type> {
    let mut ret_type = Type::Unknown;
    local_scopes.start_subscope();
    

    while *cursor != until {
        let expr_i = *cursor;
        let expr = exprs[expr_i];

        // TODO: switch to check for statements,
        // call check_expr in else branch
        match expr {
            Expr::Nop => anyhow::bail!("nop has no type"),
            Expr::Int(_) => {
                assert_eq!(types[expr_i], Type::UInt64);
                // FIXME: check type conflicts
                ret_type = Type::UInt64;
            },
            Expr::String(_) => {
                assert_eq!(types[expr_i], Type::String);
                // FIXME: check type conflicts
                ret_type = Type::String;
            },
            Expr::Ident(name) => {
                // TODO: implicit returns?
                let ref_i = local_scopes.get(name).with_context(||
                    // FIXME: need way to get name from DIndex here
                    format!("unbound identifier: {:?}", name)
                )?;
                ret_type = types[ref_i];
            },
            Expr::Binop { op, lhs, rhs } => {
                check_binop(&exprs, types, refs, cursor, expr_i, op, lhs, rhs)?;
            }
            Expr::If { cond, branch_true, branch_false } => {
                let Type::Bool = check_expr(exprs, types, refs, cursor, cond)? else {
                    anyhow::bail!("expected bool in if condition");
                };

                let branch_true_type = check_block(exprs, types, refs, extra, cursor, module_scopes, local_scopes, branch_true.end_i())?;
                let branch_false_type = check_block(exprs, types, refs, extra, cursor, module_scopes, local_scopes, branch_false.end_i())?;

                if branch_true_type != branch_false_type {
                    anyhow::bail!("mismatched types in if branches");
                }

                ret_type = branch_true_type;
            },
            Expr::Bind { .. } => todo!("bind"),
            Expr::FunCall { .. } => todo!("fun call"),
            Expr::FunArg => anyhow::bail!("expected block level node found fun arg"),
            Expr::FunDef {..} => anyhow::bail!("expected block level node found fun def"),
        }
        *cursor += 1;
        if *cursor == exprs.len() {
            panic!("block end not found");
        }
    }
    *cursor += 1;
    if ret_type == Type::Unknown {
        ret_type = Type::Void;
    }
    local_scopes.end_subscope();
    return Ok(ret_type);
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
        crate::parser::Parser::new(contents).parse()
    }

    // TODO: make macros for checking that the typecheck fails with specific errors
    macro_rules! assert_accepts {
        ($src:expr) => {
            {
                let mut ast = parse($src).expect("parser error");
                // ast.print();
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
                // ast.print();
                dbg!(&ast.exprs);
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
    fn wrong_return_type() {
        assert_rejects!("(fun foo ():str 1)");
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

    #[test]
    fn incorrect_args() {
        assert_rejects!(r#"(fun foo (a: int): int a) (fun bar () (foo "str"))"#);
    }

    #[test]
    fn correct_args() {
        assert_accepts!(r#"(fun foo (a: int): int a) (fun bar () (foo 1))"#);
    }
}
