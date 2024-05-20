use anyhow::{Context, Result};

use super::{Ast, DIndex, EIndex, Expr, Expr::*, Extra, RefIdent, Type};
use crate::{ast, parser::XIndex};

pub fn typecheck(ast: &mut Ast) -> Result<()> {
    let mut i = 0;

    let mut scopes = Scopes {
        locals: ScopeStack::new(),
        module: ModuleScopes::from(&ast.exprs, &ast.extra),
    };

    while i < ast.exprs.len() {
        let expr_i = i;
        let expr = ast.exprs[expr_i];

        let ctx = Ctx {
            exprs: &ast.exprs,
            types: &mut ast.types,
            refs: &mut ast.refs,
            extra: &ast.extra,
            cursor: &mut i,
            scopes: &mut scopes,
            data: &ast.data,
        };

        // TODO: switch to check for statements,
        // call check_expr in else branch
        match expr {
            Expr::Nop => anyhow::bail!("nop has no type"),
            Expr::Int(_) | Expr::String(_) | Expr::Ident(_) | Expr::Bool(_) => {
                let mut s = "".to_string();
                let value = match expr {
                    Expr::Int(value) => {
                        s = ast.get_const::<u64>(value).to_string();
                        &s
                    },
                    Expr::String(value) | Expr::Ident(value) => ast.get_ident(value),
                    Expr::Bool(b) => if b { "true" } else { "false" },
                    _ => unreachable!(),
                };
                anyhow::bail!("top level literal `{}` at {} not valid. malformed AST", value, i)
            }
            Expr::Return { .. } => {
                anyhow::bail!("return outside of function")
            }
            Expr::Binop { op, lhs, rhs } => {
                check_binop(&ctx, op, lhs, rhs)?;
            }
            Expr::FunDef { name, args, body } => {
                let Ctx {
                    exprs,
                    types,
                    extra,
                    cursor: i,
                    scopes,
                    ..
                } = ctx.clone();
                let fun_i = *i;
                debug_assert_eq!(scopes.module.functions.find(name), Some(*i));

                *i += 1;
                let args_slice = extra.slice(args);
                for &arg in args_slice {
                    assert!(exprs[*i] == FunArg, "expected fun arg");
                    scopes.locals.bind(arg as usize, *i);
                    *i += 1;
                }
                // debug_assert_eq!(*i, extra.slice(body).first().map(|&n| n as usize).unwrap_or(*i));
                let (res_type, _) = check_block(&ctx, body)?;
                scopes.locals.clear();
                let ret_type = &mut types[fun_i];
                if *ret_type == Type::Unknown {
                    *ret_type = res_type;
                } else if res_type != types[fun_i] {
                    anyhow::bail!("function return type does not match actual return type");
                }
            }
            Expr::If {..} => todo!("if"),
            Expr::While { cond, body } => {
                let Type::Bool = check_expr(&ctx, cond)? else {
                    anyhow::bail!("expected bool in while condition");
                };
                check_block(&ctx, body)?;
            }
            // Expr::If {
            //     cond,
            //     branch_then: branch_true,
            //     branch_else: branch_false,
            // } => {
            //     let Type::Bool = check_expr(&ctx, cond)? else {
            //         anyhow::bail!("expected bool in if condition");
            //     };
            //
            //     let branch_true_type = check_block(&ctx, branch_true)?;
            //     let branch_false_type = check_block(&ctx, branch_false)?;
            //
            //     if branch_true_type != branch_false_type {
            //         anyhow::bail!("mismatched types in if branches");
            //     }
            // }
            Expr::Bind { name, value } => {
                debug_assert_eq!(ctx.scopes.module.globals.find(name), Some(value));
                let ty = check_expr(&ctx, value)?;
                // TODO: mark visited even if type is known
                // this should be:
                // if (type is unknown)
                //  { set type = check_expr (marking expr visited)}
                //  else {mark expr visited}
                if ctx.types[value] != ty {
                    ctx.types[value] = ty;
                }
            },
            Expr::FunCall { name, args } => {
                check_funcall(&ctx, name, args)?;
            }
            Expr::FunArg => anyhow::bail!("expected top level node found fun arg"),
            Expr::Assign { name, value } => {
                let ref_i = ctx.scopes.module.globals.find(name).with_context(|| "unbound identifier")?;
                let ty = check_expr(&ctx, value)?;
                if ty != ctx.types[ref_i] {
                    anyhow::bail!("mismatched types in assignment");
                }
                ctx.types[expr_i] = ty;
            },
            Expr::Print { value } => {
                let ty = check_expr(&ctx, value)?;
                if ty != Type::String {
                    anyhow::bail!("expected string in print");
                }
                ctx.types[expr_i] = Type::Void;
            }
        }
        i += 1;
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum ReturnPath {
    Naked,
    Gib,
    Return,
}

fn check_block(ctx: &Ctx<'_>, block_i: XIndex) -> Result<(Type, ReturnPath)> {
    let Ctx {
        exprs,
        types,
        cursor,
        scopes,
        extra,
        ..
    } = ctx.clone();
    scopes.locals.start_subscope();

    let mut ret_path = ReturnPath::Naked;
    let mut ret_type = Type::Unknown;
    for (expr_i, &expr) in extra.indexed_iter_of(block_i, exprs) {
        // TODO: switch to check for statements,
        // call check_expr in else branch
        match expr {
            Expr::Nop => anyhow::bail!("nop has no type"),
            Expr::Int(_) => {
                assert_eq!(types[expr_i], Type::UInt64);
                // FIXME: check type conflicts
                ret_type = Type::UInt64;
            }
            Expr::String(_) => {
                assert_eq!(types[expr_i], Type::String);
                // FIXME: check type conflicts
                ret_type = Type::String;
            }
            Expr::Ident(name) => {
                // TODO: implicit returns?
                let ref_i = scopes.locals.get(name).with_context(||
                    // FIXME: need way to get name from DIndex here
                    format!("unbound identifier: {:?}", name))?;
                ret_type = types[ref_i];
            }
            Expr::Bool(_) => {
                assert_eq!(types[expr_i], Type::Bool);
                ret_type = Type::Bool;
            }
            Expr::Binop { op, lhs, rhs } => {
                ret_type = check_binop(ctx, op, lhs, rhs)?;
                types[expr_i] = ret_type;
            }
            Expr::If {
                cond,
                branch_then: branch_true,
                branch_else: branch_false,
            } => {
                let Type::Bool = check_expr(ctx, cond)? else {
                    anyhow::bail!("expected bool in if condition");
                };

                let branch_true_type = check_block(ctx, branch_true)?;
                let branch_false_type = check_block(ctx, branch_false)?;

                if branch_true_type != branch_false_type {
                    anyhow::bail!("mismatched types in if branches: lhs={:?} rhs={:?}", branch_true_type, branch_false_type);
                }

                ret_type = branch_true_type.0;
            }
            Return { value } => {
                // FIXME: give ptr to ret type as param to this function
                // so nested returns can check against it
                ret_type = match value {
                    Some(value) => check_expr(ctx, value.into())?,
                    None => Type::Void,
                };
                ret_path = ReturnPath::Return;
                // TODO: consider stopping typechecking past this point
                // note - this will require figuring out the last node
                // and marking it as visited despite not being checked
            }
            Expr::Bind { name, value } => {
                let annotated_ty = types[expr_i];
                let expr_ty = check_expr(ctx, value)?;
                if annotated_ty == Type::Unknown {
                    if expr_ty == Type::Unknown {
                        anyhow::bail!("cannot infer type of unannotated binding");
                    }
                    types[expr_i] = expr_ty;
                } else if annotated_ty != expr_ty {
                    anyhow::bail!("mismatched types in assignment");
                }
                types[expr_i] = expr_ty;
                scopes.locals.bind(name, expr_i);
            },
            Expr::FunCall { name, args } => {
                ret_type = check_funcall(&ctx, name, args)?;
            }
            Expr::FunArg => anyhow::bail!("expected block level node found fun arg"),
            Expr::FunDef { .. } => anyhow::bail!("expected block level node found fun def"),
            Expr::While { cond, body } => {
                let Type::Bool = check_expr(ctx, cond)? else {
                anyhow::bail!("expected bool in while condition");
                };
                (ret_type, _) = check_block(ctx, body)?;
            },
            Expr::Assign { name, value } => {
                let ref_i = scopes.locals.get(name).with_context(|| "unbound identifier")?;
                let ty = check_expr(ctx, value)?;
                if ty != types[ref_i] {
                    anyhow::bail!("mismatched types in assignment");
                }
                types[expr_i] = ty;
            },
            Expr::Print { value } => {
                let ty = check_expr(ctx, value)?;
                if ty != Type::String {
                    anyhow::bail!("expected string in print");
                }
                types[expr_i] = Type::Void;
            }
        }
        mark_visited(cursor, expr_i);
    }
    if ret_type == Type::Unknown {
        ret_type = Type::Void;
    }
    scopes.locals.end_subscope();
    return Ok((ret_type, ret_path));
}

fn check_expr(ctx: &Ctx<'_>, expr_i: EIndex) -> Result<Type> {
    let Ctx {
        exprs,
        types,
        cursor,
        refs,
        ..
    } = ctx.clone();
    let expr = exprs[expr_i];
    match expr {
        Expr::Nop => anyhow::bail!("nop has no type"),
        Expr::Int(_) | Expr::String(_) | Expr::Bool(_) => {
            mark_visited(cursor, expr_i);
            Ok(types[expr_i])
        }
        Expr::Binop { op, lhs, rhs } => check_binop(ctx, op, lhs, rhs),
        Expr::Bind { .. } => todo!("bind"),
        Expr::Ident(value) => {
            let ref_i = ctx.scopes.find(value).with_context(|| "unbound identifier")?;
            refs[expr_i] = Some(ref_i);
            let ty = types[ref_i];
            types[expr_i] = ty;
            mark_visited(cursor, expr_i);
            Ok(ty)
        },
        Expr::If { cond, branch_then, branch_else } => {
                let Type::Bool = check_expr(ctx, cond)? else {
                    anyhow::bail!("expected bool in if condition");
                };

                let branch_true_type = check_block(ctx, branch_then)?;
                let branch_false_type = check_block(ctx, branch_else)?;

                if branch_true_type != branch_false_type {
                    anyhow::bail!("mismatched types in if branches");
                }
                Ok(branch_true_type.0)
        },
        Expr::Print {value} => {
            let ty = check_expr(ctx, value)?;
            if ty != Type::String {
                anyhow::bail!("expected string in print");
            }
            Ok(Type::Void)
        },
        Expr::FunCall { name, args } => check_funcall(ctx, name, args),
        Expr::FunDef { .. } => todo!("fun def"),
        Expr::FunArg => anyhow::bail!("expected expr found fun arg"),
        Expr::Return { .. } => anyhow::bail!("return not an expression"),
        Expr::While {..} => anyhow::bail!("while not an expression"),
        Expr::Assign { ..} => anyhow::bail!("assign not an expression"),
    }
}

fn check_binop(ctx: &Ctx<'_>, op: ast::Binop, lhs_i: EIndex, rhs_i: EIndex) -> Result<Type> {
    let Ctx { types, cursor, .. } = ctx.clone();
    let binop_i = *cursor;
    let lhs_type = check_expr(ctx, lhs_i)?;
    let rhs_type = check_expr(ctx, rhs_i)?;
    if lhs_type != rhs_type {
        // TODO: better error message
        anyhow::bail!("mismatched types in binop");
    }
    let binop_type = match op {
        ast::Binop::Add | ast::Binop::Sub | ast::Binop::Mul | ast::Binop::Div | ast::Binop::Mod => match lhs_type {
            Type::UInt64 => lhs_type,
            _ => anyhow::bail!("invalid type for arithmetic op"),
        },
        ast::Binop::And => match lhs_type {
            Type::Bool => Type::Bool,
            _ => anyhow::bail!("invalid type for logical op"),
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

fn check_funcall(ctx: &Ctx<'_>, name: DIndex, args: XIndex) -> Result<Type> {
    let Ctx {
        exprs,
        types,
        extra,
        cursor,
        scopes,
        data,
        ..
    } = ctx.clone();
    let fun_i = scopes
        .module
        .functions
        .find(dbg!(name))
        .with_context(|| format!("could not find function {}", data.get_ref::<str>(name)))?;
    let Expr::FunDef { args: params_i, .. } = exprs[fun_i] else {
        anyhow::bail!("expected fun def");
    };
    mark_visited(cursor, fun_i);
    let num_params = extra.len_of(params_i);
    let param_types_range = fun_i + 1..=(fun_i + num_params);
    for (arg, param_type_i) in extra.iter(args).zip(param_types_range) {
        let arg = arg as usize;
        let arg_type = check_expr(ctx, arg)?;
        let param_type = types[param_type_i];
        if arg_type != param_type {
            anyhow::bail!("mismatched types in function call");
        }
    }
    mark_visited(cursor, extra.last_of(args) as usize);

    Ok(types[fun_i])
}

fn mark_visited(cursor: &mut usize, i: usize) {
    *cursor = usize::max(*cursor, i);
}

struct Ctx<'a> {
    exprs: &'a [Expr],
    types: &'a mut [Type],
    refs: &'a mut [RefIdent],
    extra: &'a Extra,
    cursor: &'a mut usize,
    scopes: &'a mut Scopes,
    data: &'a ast::DataPool,
}

impl Clone for Ctx<'_> {
    fn clone(&self) -> Self {
        Self {
            exprs: self.exprs,
            types: unsafe { &mut *(self.types as *const [Type] as *mut [Type]) },
            refs: unsafe { &mut *(self.refs as *const [RefIdent] as *mut [RefIdent]) },
            extra: self.extra,
            cursor: unsafe { &mut *(self.cursor as *const usize as *mut usize) },
            scopes: unsafe { &mut *(self.scopes as *const Scopes as *mut Scopes) },
            data: self.data,
        }
    }
}

#[derive(Debug)]
struct Scopes {
    locals: ScopeStack,
    module: ModuleScopes,
}

impl Scopes {
    fn find(&self, name: DIndex) -> Option<EIndex> {
        self.locals.get(name).or_else(|| self.module.globals.find(name))
    }
}

/// Collection of scopes related to a module
#[derive(Debug)]
struct ModuleScopes {
    globals: ScopeList,
    /// also globals, but not variables
    functions: ScopeList,
}

impl ModuleScopes {
    pub fn from(exprs: &[Expr], extra: &ast::Extra) -> Self {
        let mut globals = ScopeList::new();
        let mut last_fundef_end = 0;

        let mut functions = ScopeList::new();

        // PERF: create a vec of info then sort instead of inserting
        // one by one
        for (i, expr) in exprs.iter().enumerate() {
            match expr {
                FunDef { name, body, .. } => {
                    last_fundef_end = extra.last_of(*body) as usize;
                    functions.insert(*name, i).expect("function name already bound");
                }
                Bind { name, value } => {
                    if i >= last_fundef_end {
                        globals.insert(*name, *value).unwrap();
                    }
                }
                _ => continue,
            }
        }

        Self { globals, functions }
    }
}

#[derive(Debug)]
struct ScopeList {
    indices: Vec<EIndex>,
    names: Vec<DIndex>,
}

impl ScopeList {
    fn new() -> Self {
        Self {
            indices: vec![],
            names: vec![],
        }
    }
    fn new_from(names: Vec<DIndex>, indices: Vec<EIndex>) -> Self {
        debug_assert!(
            {
                let mut sorted = true;
                let mut last: DIndex = 0;
                for &name in &names {
                    if name > last {
                        sorted = false;
                        break;
                    }
                    last = name;
                }
                sorted
            },
            "names not sorted"
        );
        Self { names, indices }
    }
    fn find(&self, name: DIndex) -> Option<EIndex> {
        let index = self.names.binary_search(&name).ok()?;
        Some(self.indices[index])
    }
    fn insert(&mut self, name: DIndex, expr_i: EIndex) -> Result<()> {
        let index = match self.names.binary_search(&name) {
            /* binary search returns the index to insert at if not found */
            Err(index) => index,
            Ok(_) => anyhow::bail!("name already bound"),
        };
        self.names.insert(index, name);
        self.indices.insert(index, expr_i);
        Ok(())
    }
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
        ($src:expr) => {{
            let mut ast = parse($src).expect("parser error");
            ast.print();
            let res = super::typecheck(&mut ast);
            assert!(res.is_ok(), "{}", res.unwrap_err());
            ast
        }};
    }

    macro_rules! assert_rejects {
        ($src:expr) => {{
            let mut ast = parse($src).expect("parser error");
            // ast.print();
            // dbg!((0..).zip(&ast.exprs).collect::<Vec<_>>());
            let res = super::typecheck(&mut ast);
            assert!(res.is_err());
            ast
        }};
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
        assert_rejects!(r#"fun foo () 1 (== foo() "some_str")"#);
    }

    #[test]
    fn multiple_fun_return_type_misused() {
        // FIXME: this test is passing but idk why (probably because it sees unknown as invalid
        // type)
        // but this isn't right! Should make `assert_rejects_with` as well as typecheck error union
        // type and check that error matches pattern
        assert_rejects!(
            r#"
            fun foo () {return 1}
            fun bar() {return "str"}
            (== foo() bar())
        "#
        );
    }

    #[test]
    fn wrong_return_type() {
        assert_rejects!("fun foo () str { return 1}");
    }

    #[test]
    fn mismatched_branch_arms() {
        assert_rejects!(r#"let foo str = if (== 1 2) 1 else "str""#);
    }

    #[test]
    fn mismatched_return_types() {
        assert_rejects!(r#"fun foo() { if (== 1 2) 1 else "str"}"#);
    }

    #[test]
    fn fun_arg_is_return_type() {
        assert_accepts!(r#"fun foo (a int) int {return a}"#);
    }

    #[test]
    fn fun_call() {
        assert_accepts!(r#"fun foo (a int) int {return a} foo(10)"#);
    }

    #[test]
    fn unbound_ident() {
        assert_rejects!("fun foo () { return a }");
    }

    #[test]
    fn incorrect_args() {
        assert_rejects!(r#"fun foo (a int) int a fun bar () {return foo("str")}"#);
    }

    #[test]
    fn correct_args() {
        assert_accepts!(r#"fun foo(a int) int {return a} fun bar() {return foo(1)}"#);
    }
}
