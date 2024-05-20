use anyhow::{Context, Result};

use crate::ast::{Ast, DIndex, Range};
use crate::parser::{Binop, EIndex, Expr, XIndex};

#[derive(Debug, Clone, Copy)]
pub enum ByteCode {
    Nop,
    Push(u64),
    Mov(u32),
    Jump(u32),
    JumpIfZero(u32),
    Store(u32),
    Load(u32),
    Call(u32),
    Ret,
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Not,
    Mod,
    And,
    Print,
}

impl From<Binop> for ByteCode {
    fn from(op: Binop) -> Self {
        match op {
            Binop::Add => ByteCode::Add,
            Binop::Sub => ByteCode::Sub,
            Binop::Mul => ByteCode::Mul,
            Binop::Div => ByteCode::Div,
            Binop::Eq => ByteCode::Eq,
            Binop::Lt => ByteCode::Lt,
            Binop::LtEq => ByteCode::LtEq,
            Binop::Gt => ByteCode::Gt,
            Binop::GtEq => ByteCode::GtEq,
            Binop::Mod => ByteCode::Mod,
            Binop::And => ByteCode::And,
        }
    }
}

#[derive(Debug)]
pub struct Chunk {
    pub ops: Vec<ByteCode>,
    pub data: Vec<u8>,
}

pub struct Compiler {
    ast: Ast,
    bytecode: Chunk,
    expr_i: usize,
    visited: Vec<bool>,
    scope_stack: ScopeStack,
    fun_map: FunMap,
}

impl Compiler {
    pub fn new(ast: Ast) -> Self {
        let visited = vec![false; ast.exprs.len()];
        Self {
            ast,
            bytecode: Chunk {
                ops: Vec::new(),
                data: Vec::new(),
            },
            expr_i: 0,
            visited,
            scope_stack: ScopeStack::new(),
            fun_map: FunMap::new(),
        }
    }

    fn next_expr(&mut self) -> Option<Expr> {
        let num_exprs = self.ast.exprs.len();

        // skip visited
        // TODO: is there a way to know what has been visited based on tree structure?
        while self.expr_i < num_exprs && self.visited[self.expr_i] {
            self.expr_i += 1;
        }
        if self.expr_i >= self.ast.exprs.len() {
            return None;
        }
        let expr = self.ast.exprs[self.expr_i];
        self.mark_visited(self.expr_i);
        self.expr_i += 1;
        Some(expr)
    }

    fn dbg_visited(&self) {
        let visited = self.visited.iter().zip(&self.ast.exprs).collect::<Vec<_>>();
        dbg!(visited);
    }

    fn mark_visited(&mut self, expr_i: usize) {
        self.visited[expr_i] = true;
    }

    pub fn compile(&mut self) {
        while let Some(expr) = self.next_expr() {
            self._compile_expr(expr);
        }
    }

    fn _compile_expr(&mut self, expr: Expr) {
        match expr {
            Expr::Int(i) => {
                self.compile_int(i);
            }
            Expr::Binop { op, lhs, rhs } => {
                self.compile_binop(op, lhs, rhs);
            }
            Expr::Bool(b) => {
                self.emit(ByteCode::Push(if b { 1 } else { 0 }));
            }
            Expr::If {
                cond,
                branch_then: branch_true,
                branch_else: branch_false,
            } => {
                self.compile_if(cond, branch_true, branch_false);
            }
            Expr::Bind { name, value } => {
                self.compile_bind(name, value);
            }
            Expr::Ident(name) => {
                self.compile_load(name).expect("compile load failed");
            }
            Expr::String(value) => {
                let data = self.ast.data.get_ref::<[u8]>(value);
                let data_i = self.bytecode.data.len() as u64;
                self.bytecode.data.extend_from_slice(data);
                self.bytecode.data.push(0);
                self.emit(ByteCode::Push(data_i));
            }
            Expr::FunDef { name, args, body } => {
                self.compile_fundef(name, args, body);
            }
            Expr::FunCall { name, args } => {
                self.compile_fun_call(name, args)
                    .expect("compile fun call failed");
            }
            Expr::Return { value } => match value {
                Some(value) => {
                    self.compile_expr(usize::from(value));
                    self.emit(ByteCode::Ret);
                }
                None => {
                    self.emit(ByteCode::Push(0));
                    self.emit(ByteCode::Ret);
                }
            },
            Expr::While { cond, body } => {
                self.compile_expr(cond);
                self.emit(ByteCode::Push(0));
                self.emit(ByteCode::Not);
                let jmp_start = self.init_jmp();

                let while_body_start = self.current_offset();
                self.compile_block(body);

                self.compile_expr(cond);
                self.emit(ByteCode::JumpIfZero(while_body_start as u32));
                self.end_jmp_if_zero(jmp_start)
            }
            Expr::Print { value } => {
                self.compile_expr(value);
                let type_code = match self.ast.types[value] {
                    crate::ast::Type::UInt64 => Type::Int,
                    crate::ast::Type::Bool => Type::Bool,
                    crate::ast::Type::String => Type::String,
                    crate::ast::Type::Void => unreachable!("void type"),
                    crate::ast::Type::Unknown => unreachable!("unknown type"),
                };
                self.emit(ByteCode::Push(type_code.as_immediate()));
                self.emit(ByteCode::Print);
            }
            Expr::Assign { name, value } => {
                self.compile_expr(value);
                let i = self.scope_stack.get(name).expect("variable not found");
                self.emit(ByteCode::Store(i));
            }
            _ => unimplemented!("Expr: {:?} not implemented", expr),
        }
    }

    fn compile_expr(&mut self, i: EIndex) {
        self.mark_visited(i);
        self._compile_expr(self.ast.exprs[i]);
    }

    fn compile_binop(&mut self, op: Binop, lhs: EIndex, rhs: EIndex) {
        self.compile_expr(lhs);
        self.compile_expr(rhs);
        // FIXME:
        let bc_op = ByteCode::from(op);
        self.emit(bc_op);
    }

    fn compile_if(&mut self, cond: EIndex, branch_true: XIndex, branch_false: XIndex) {
        self.compile_expr(cond);

        let jmp_true_i = self.init_jmp();
        let jmp_else_i = self.init_jmp();

        self.end_jmp_if_zero(jmp_true_i);

        self.compile_block(branch_true);

        let jmp_end_i = self.reserve();

        self.end_jmp(jmp_else_i);

        self.compile_block(branch_false);
        self.end_jmp(jmp_end_i);
    }

    fn compile_block(&mut self, block: XIndex) {
        let exprs = self.ast.extra.slice(block);
        for expr_i in exprs {
            crate::utils::steal!(Self, self).compile_expr(*expr_i as usize);
        }
    }

    fn compile_bind(&mut self, name: DIndex, value: EIndex) {
        self.compile_expr(value);
        let i = self.scope_stack.get(name).expect("variable not found");
        self.emit(ByteCode::Store(i));
    }

    fn compile_load(&mut self, name: DIndex) -> Result<()> {
        let i = self.scope_stack.get(name).with_context(|| {
            let name = self.ast.data.get_ref::<str>(name);
            format!("variable not found: {}", name)
        })?;
        dbg!((self.ast.data.get_ref::<str>(name), ":=", i));
        self.emit(ByteCode::Load(i));
        Ok(())
    }

    fn set(&mut self, i: usize, bc: ByteCode) {
        self.bytecode.ops[i] = bc;
    }

    fn emit(&mut self, bc: ByteCode) {
        self.bytecode.ops.push(bc);
    }

    fn current_offset(&self) -> usize {
        self.bytecode.ops.len()
    }

    fn reserve(&mut self) -> usize {
        let i = self.current_offset();
        self.emit(ByteCode::Nop);
        return i;
    }

    fn init_jmp(&mut self) -> usize {
        return self.reserve();
    }

    fn end_jmp_if_zero(&mut self, jmp_i: usize) {
        let len = self.current_offset();
        let jump_offset = len as u32;
        self.set(jmp_i, ByteCode::JumpIfZero(jump_offset));
    }

    fn end_jmp(&mut self, jmp_i: usize) {
        let len = self.current_offset();
        let jump_offset = len as u32;
        self.set(jmp_i, ByteCode::Jump(jump_offset));
    }

    fn compile_int(&mut self, di: DIndex) {
        let val: u64 = self.ast.get_const(di);
        self.emit(ByteCode::Push(val));
    }

    pub fn bytecode(&self) -> &Chunk {
        &self.bytecode
    }

    fn compile_fundef(&mut self, name: DIndex, args: EIndex, body: XIndex) {
        let start = self.init_jmp();
        self.fun_map.add(name, self.current_offset() as u32, args);
        self.scope_stack.start_new();
        self.bind_fun_args(args);
        self.allocate_locals(body);
        self.compile_block(body);
        self.emit(ByteCode::Ret);
        self.end_jmp(start);
        self.scope_stack.end();
    }

    fn bind_fun_args(&mut self, args_start: usize) {
        let ast = &self.ast;
        let fun_args = ast.extra.slice(args_start);
        let num_args = fun_args.len();
        for &arg_name in fun_args {
            self.scope_stack.bind_local(arg_name as DIndex);
        }
        // separated because rust
        for _ in 0..num_args {
            let Some(Expr::FunArg) = self.next_expr() else {
                unreachable!("expected fun arg");
            };
        }
    }

    fn allocate_locals(&mut self, body: XIndex) {
        let body_exprs = self.ast.extra.slice(body);
        if body_exprs.is_empty() {
            return;
        }
        let first = *body_exprs.first().expect("first") as usize;
        let mut last = *body_exprs.last().expect("last") as usize;
        loop {
            let last_expr = self.ast.exprs[last];
            match last_expr {
                Expr::Nop | Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::Ident(_) => {
                    break;
                }
                Expr::Bind { value, .. } | Expr::Assign { value, .. } | Expr::Print { value } => {
                    last = value;
                }
                Expr::If {
                    cond,
                    branch_then,
                    branch_else,
                } => {
                    last = self
                        .ast
                        .extra
                        .last_of(branch_else)
                        .map(|x| x as usize)
                        .unwrap_or_else(|| {
                            self.ast
                                .extra
                                .last_of(branch_then)
                                .map(|x| x as usize)
                                .unwrap_or(cond)
                        });
                }
                Expr::Binop { rhs, .. } => last = rhs,
                Expr::FunCall { args, .. } => {
                    last = self
                        .ast
                        .extra
                        .last_of(args)
                        .map(|x| x as usize)
                        .unwrap_or(last);
                }
                Expr::Return { value } => {
                    if let Some(value) = value {
                        last = usize::from(value);
                    } else {
                        break;
                    }
                }
                Expr::While { cond, body } => {
                    last = self
                        .ast
                        .extra
                        .last_of(body)
                        .map(|x| x as usize)
                        .unwrap_or(cond);
                }
                Expr::FunArg => unreachable!("fun arg"),
                Expr::FunDef { .. } => unreachable!("function definition in block"),
            }
        }
        for i in first..=last {
            let Expr::Bind {name, ..} = self.ast.exprs[i] else {
                continue;
            };
            self.scope_stack.bind_local(name);
            self.emit(ByteCode::Push(0));
        }
    }

    fn compile_fun_call(&mut self, name: DIndex, args_start: EIndex) -> Result<()> {
        let fun_offset = self.fun_map.get_offset(name).with_context(|| {
            let name = self.ast.data.get_ref::<str>(name);
            format!("function not found: {}", name)
        })?;

        // using args_range here because rust cannot figure out
        // that the immutable borrow of args with `ast::fun_args_slice()`
        // is not mutated by `self.compile_expr()`
        let args_range = self.ast.args_range(args_start);
        let num_args = args_range.len();

        for arg_i in args_range {
            let arg = self.ast.extra[arg_i];
            self.compile_expr(arg as EIndex);
        }
        self.emit(ByteCode::Push(num_args as u64));
        self.emit(ByteCode::Call(fun_offset));
        Ok(())
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Type {
    Int,
    Bool,
    String
}

impl Type {
    pub fn as_immediate(&self) -> u64 {
        *self as u64
    }

    pub fn from_bc(bc: i64) -> Self {
        let lut = [Self::Int, Self::Bool, Self::String];
        return lut[bc as usize];
    }
}

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

    // TODO: remove
    fn skip<const N: usize>(&mut self) {
        let vals = [Self::NOT_BOUND; N];
        self.stack_map.extend(vals);
    }

    /// like start_new but does not set `cur`
    // TODO: remove
    fn start_subscope(&mut self) {
        self.starts.push(self.stack_map.len());
    }

    // TODO: remove
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

    // [ 0 1 2 3 4]
    fn get(&self, name: DIndex) -> Option<u32> {
        debug_assert_ne!(name, Self::NOT_BOUND);
        let pos = self.stack_map.iter().rev().position(|&n| n == name);
        let Some(pos) = pos else {
            return None;
        };
        let pos = self.stack_map.len() - pos - 1;
        if pos < self.cur {
            unimplemented!("globals not implemented");
        }
        return Some((pos - self.cur) as u32);
    }

    // TODO: remove
    fn is_last(&self, i: u32) -> bool {
        let i = self.cur + i as usize;
        let len = self.stack_map.len();
        return i + 1 == len;
    }
}

struct FunMap {
    offsets: Vec<u32>,
    arg_starts: Vec<usize>,
    names: Vec<DIndex>,
}

impl FunMap {
    fn new() -> Self {
        Self {
            offsets: vec![],
            names: vec![],
            arg_starts: vec![],
        }
    }

    fn add(&mut self, name: DIndex, offset: u32, arg_start: usize) {
        self.names.push(name);
        self.offsets.push(offset);
        self.arg_starts.push(arg_start);
    }

    fn get_offset(&self, name: DIndex) -> Option<u32> {
        self.names
            .iter()
            .position(|&n| n == name)
            .map(|i| self.offsets[i])
    }

    #[allow(unused)]
    fn get_arg_start(&self, name: DIndex) -> Option<usize> {
        self.names
            .iter()
            .position(|&n| n == name)
            .map(|i| self.arg_starts[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ByteCode::*;
    use crate::parser::tests::assert_matches;

    mod scope_stack {
        use super::*;

        #[test]
        fn is_last() {
            let mut s = ScopeStack::new();
            let name = 13;
            let i = s.bind_local(name);
            let found = s.get(name).expect("found");
            assert_eq!(found, i);
            assert!(s.is_last(found));
        }
    }

    fn compile(contents: &str) -> Compiler {
        let parser = crate::parser::Parser::new(contents);
        let mut compiler = Compiler::new(parser.parse().expect("syntax error"));
        compiler.compile();
        return compiler;
    }

    macro_rules! assert_bytecode_matches {
        ($bytecode:expr, [$($ops:pat),*]) => {
            let mut i = 0;
            $(
                #[allow(unused_assignments)]
                {
                    assert_matches!($bytecode.ops[i], $ops);
                    i += 1;
                }
            )*
            assert_eq!($bytecode.ops.len(), i, "expected {} ops, got {}. Extra: {:?}", i, $bytecode.ops.len(), &$bytecode.ops[i..]);
        };
    }

    macro_rules! assert_compiles_to {
        ($contents:expr, [$($ops:pat),*]) => {
            let contents = $contents;
            let compiler = compile(contents);
            assert_bytecode_matches!(compiler.bytecode, [$($ops),*]);
        };
    }

    #[test]
    fn add() {
        let contents = "( + 1 2 )";
        let compiler = compile(contents);
        assert_bytecode_matches!(
            compiler.bytecode,
            [ByteCode::Push(1), ByteCode::Push(2), ByteCode::Add]
        );
    }

    #[test]
    fn nested_add() {
        let contents = "( + 1 ( + 2 3 ) )";
        let compiler = compile(contents);
        assert_bytecode_matches!(compiler.bytecode, [Push(1), Push(2), Push(3), Add, Add]);
    }

    #[test]
    fn nested_add_lhs() {
        assert_compiles_to!("( + ( + 1 2 ) 3 )", [Push(1), Push(2), Add, Push(3), Add]);
    }

    #[test]
    fn repeated_add() {
        let contents = "(+ 1 2) (+ 3 4)";
        let compiler = compile(contents);
        assert_bytecode_matches!(
            compiler.bytecode,
            [Push(1), Push(2), Add, Push(3), Push(4), Add]
        );
    }

    #[test]
    fn all_binops() {
        let contents = "(+ 1 2) (- 3 4) (* 5 6) (/ 7 8)";
        let compiler = compile(contents);
        assert_bytecode_matches!(
            compiler.bytecode,
            [
                Push(1),
                Push(2),
                Add,
                Push(3),
                Push(4),
                Sub,
                Push(5),
                Push(6),
                Mul,
                Push(7),
                Push(8),
                Div
            ]
        );
    }

    #[test]
    fn if_expr() {
        let contents = "if true 2 else 3";
        let compiler = compile(contents);
        dbg!(&compiler.bytecode.ops);
        assert_bytecode_matches!(
            compiler.bytecode,
            [Push(1), JumpIfZero(3), Jump(5), Push(2), Jump(6), Push(3)]
        );
    }

    #[test]
    fn let_bind() {
        assert_compiles_to!("let x = 1", [ByteCode::Push(1)]);
    }

    #[test]
    fn let_bind_and_use() {
        assert_compiles_to!("let x = 1 (+ x 1)", [Push(1), Load(0), Push(1), Add]);
    }

    #[test]
    fn fun_def() {
        assert_compiles_to!(
            "fun foo(x int) int {return x}",
            [
                // TODO: implement placing functions somewhere
                // and storing offsets to them instead of just
                // skipping them when they are encountered
                Jump(3),
                Load(2),
                Ret // Load arg (pushing it to top of stack)
                    // return
            ]
        );
    }

    #[test]
    fn fun_call() {
        assert_compiles_to!(
            "fun foo(x int) {return x} foo(1)",
            [
                Jump(3),
                Load(2),
                Ret, // Load arg (pushing it to top of stack)
                Push(1),
                Push(1), // number of args
                Call(1)
            ]
        );
    }

    #[test]
    fn fun_call_multiple_args() {
        assert_compiles_to!(
            "fun foo(x int, y int) int {return (+ x y)} foo(1, 2)",
            [
                Jump(5),
                Load(3),
                Load(2),
                Add,
                Ret, // Load arg (pushing it to top of stack)
                Push(1),
                Push(2),
                Push(2), // number of args
                Call(1)
            ]
        );
    }

    #[test]
    fn fun_call_rec() {
        assert_compiles_to!(
            "fun foo(x int) int { return foo(( - x 1 )) } foo(10)",
            [
                Jump(7),
                Load(2),
                Push(1),
                Sub,
                Push(1),
                Call(1),
                Ret, // Load arg (pushing it to top of stack)
                Push(10),
                Push(1), // number of args
                Call(1)
            ]
        );
    }

    #[test]
    fn countdown() {
        assert_compiles_to!(
            r#"
                fun countdown (n int) int {
                    return if (> n 0)
                        countdown((- n 1))
                        else n
                }
                countdown(10)
            "#,
            [
                Jump(14),
                Load(2),
                Push(0),
                Gt,
                JumpIfZero(6),
                Jump(12),
                Load(2),
                Push(1),
                Sub,
                Push(1),
                Call(1),
                Jump(13),
                Load(2),
                Ret,
                Push(10),
                Push(1),
                Call(1)
            ]
        );
    }
}
