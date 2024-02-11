use anyhow::{Context, Result};

use crate::ast::{Ast, DIndex};
use crate::parser::{Binop, EIndex, Expr};

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
}

impl From<Binop> for ByteCode {
    fn from(op: Binop) -> Self {
        match op {
            Binop::Plus => ByteCode::Add,
            Binop::Minus => ByteCode::Sub,
            Binop::Mul => ByteCode::Mul,
            Binop::Div => ByteCode::Div,
            Binop::Eq => ByteCode::Eq,
            Binop::Lt => ByteCode::Lt,
            Binop::LtEq => ByteCode::LtEq,
            Binop::Gt => ByteCode::Gt,
            Binop::GtEq => ByteCode::GtEq,
        }
    }
}

#[derive(Debug)]
pub struct Chunk {
    pub ops: Vec<ByteCode>,
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
            bytecode: Chunk { ops: Vec::new() },
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
            Expr::If {
                cond,
                branch_true,
                branch_false,
            } => {
                self.compile_if(cond, branch_true, branch_false);
            }
            Expr::Bind { name, value } => {
                self.compile_bind(name, value);
            }
            Expr::Ident(name) => {
                self.compile_load(name).expect("compile load failed");
            }
            Expr::FunDef { name, args, body } => {
                self.compile_fundef(name, args, body);
            }
            Expr::FunCall { name, args } => {
                self.compile_fun_call(name, args)
                    .expect("compile fun call failed");
            }
            _ => unimplemented!("Expr: {:?} not implemented", expr),
        }
    }

    fn compile_expr(&mut self, i: EIndex) {
        self._compile_expr(self.ast.exprs[i]);
        self.mark_visited(i);
    }

    fn compile_binop(&mut self, op: Binop, lhs: EIndex, rhs: EIndex) {
        self.compile_expr(lhs);
        self.compile_expr(rhs);
        // FIXME:
        let bc_op = ByteCode::from(op);
        self.emit(bc_op);
    }

    fn compile_if(&mut self, cond: EIndex, branch_true: EIndex, branch_false: EIndex) {
        self.compile_expr(cond);

        let jmp_true_i = self.init_jmp();
        let jmp_else_i = self.init_jmp();

        self.end_jmp_if_zero(jmp_true_i);

        self.compile_expr(branch_true);

        let jmp_end_i = self.reserve();

        self.end_jmp(jmp_else_i);

        self.compile_expr(branch_false);
        self.end_jmp(jmp_end_i);
    }

    fn compile_bind(&mut self, name: DIndex, value: EIndex) {
        self.compile_expr(value);
        let i = self.scope_stack.bind_local(name);
        self.emit(ByteCode::Store(i));
    }

    fn compile_load(&mut self, name: DIndex) -> Result<()> {
        let i = self.scope_stack.get(name).with_context(|| {
            let name = self.ast.data.get_ref::<str>(name);
            format!("variable not found: {}", name)
        })?;
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

    fn compile_fundef(&mut self, name: DIndex, args: EIndex, body: EIndex) {
        let start = self.init_jmp();
        self.fun_map.add(name, self.current_offset() as u32, args);
        self.scope_stack.start_new();
        self.compile_fun_args(args);
        self.compile_expr(body);
        self.emit(ByteCode::Ret);
        self.end_jmp(start);
        self.scope_stack.end();
    }

    fn compile_fun_args(&mut self, args_start: usize) {
        let args_range = self.ast.args_range(args_start);
        for arg_i in args_range {
            self.mark_visited(arg_i);
            let arg = self.ast.exprs[arg_i];
            let Expr::FunArg { name, len: _ } = arg else {
                unreachable!("funarg should be a funarg");
            };
            self.scope_stack.bind_local(name);
        }
    }

    fn compile_fun_call(&mut self, name: DIndex, args_start: EIndex) -> Result<()> {
        let fun_offset = self.fun_map.get_offset(name).with_context(|| {
            let name = self.ast.data.get_ref::<str>(name);
            format!("function not found: {}", name)
        })?;

        let args = self.ast.args_range(args_start);
        let num_args = args.len() as u32;

        for arg_i in args {
            let Expr::FunCallArg { value, .. } = self.ast.exprs[arg_i] else {
                unreachable!("fun call arg should be a funcallarg");
            };
            self.compile_expr(value);
            self.mark_visited(arg_i);
        }
        self.emit(ByteCode::Push(num_args as u64));
        self.emit(ByteCode::Call(fun_offset));
        Ok(())
    }
}

struct ScopeStack {
    var_map: Vec<DIndex>,
    starts: Vec<usize>,
    cur: usize,
}

impl ScopeStack {
    fn new() -> Self {
        Self {
            var_map: vec![],
            starts: vec![0],
            cur: 0,
        }
    }

    fn start_new(&mut self) {
        self.starts.push(self.cur);
        self.cur = self.var_map.len();
    }

    fn end(&mut self) {
        let start = self.cur;
        self.cur = self.starts.pop().expect("no starts");
        self.var_map.truncate(start);
    }

    /// NOTE: does not check that name is not already in scope,
    /// therefore allowing shadowing
    fn bind_local(&mut self, name: DIndex) -> u32 {
        let i = self.var_map.len();
        self.var_map.push(name);
        return (i - self.cur) as u32;
    }

    fn get(&self, name: DIndex) -> Option<u32> {
        let pos = self.var_map
            .iter()
            .rev()
            .position(|&n| n == name);
        let Some(pos) = pos else {
            return None;
        };
        if pos < self.cur {
            unimplemented!("globals not implemented");
        }
        return Some((pos - self.cur) as u32);
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
    use crate::parser::tests::assert_matches;

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
        assert_bytecode_matches!(
            compiler.bytecode,
            [
                ByteCode::Push(1),
                ByteCode::Push(2),
                ByteCode::Push(3),
                ByteCode::Add,
                ByteCode::Add
            ]
        );
    }

    #[test]
    fn repeated_add() {
        let contents = "(+ 1 2) (+ 3 4)";
        let compiler = compile(contents);
        assert_bytecode_matches!(
            compiler.bytecode,
            [
                ByteCode::Push(1),
                ByteCode::Push(2),
                ByteCode::Add,
                ByteCode::Push(3),
                ByteCode::Push(4),
                ByteCode::Add
            ]
        );
    }

    #[test]
    fn all_binops() {
        let contents = "(+ 1 2) (- 3 4) (* 5 6) (/ 7 8)";
        let compiler = compile(contents);
        assert_bytecode_matches!(
            compiler.bytecode,
            [
                ByteCode::Push(1),
                ByteCode::Push(2),
                ByteCode::Add,
                ByteCode::Push(3),
                ByteCode::Push(4),
                ByteCode::Sub,
                ByteCode::Push(5),
                ByteCode::Push(6),
                ByteCode::Mul,
                ByteCode::Push(7),
                ByteCode::Push(8),
                ByteCode::Div
            ]
        );
    }

    #[test]
    fn if_expr() {
        let contents = "(if 1 2 3)";
        let compiler = compile(contents);
        dbg!(&compiler.bytecode.ops);
        assert_bytecode_matches!(
            compiler.bytecode,
            [
                ByteCode::Push(1),
                ByteCode::JumpIfZero(3),
                ByteCode::Jump(5),
                ByteCode::Push(2),
                ByteCode::Jump(6),
                ByteCode::Push(3)
            ]
        );
    }

    #[test]
    fn let_bind() {
        assert_compiles_to!("(let x 1)", [ByteCode::Push(1), ByteCode::Store(0)]);
    }

    #[test]
    fn let_bind_and_use() {
        assert_compiles_to!(
            "(let x 1) x",
            [ByteCode::Push(1), ByteCode::Store(0), ByteCode::Load(0)]
        );
    }

    #[test]
    fn fun_def() {
        assert_compiles_to!(
            "(fun foo (x) x)",
            [
                // TODO: implement placing functions somewhere
                // and storing offsets to them instead of just
                // skipping them when they are encountered
                ByteCode::Jump(3),
                ByteCode::Load(0),
                ByteCode::Ret // Load arg (pushing it to top of stack)
                              // return
            ]
        );
    }

    #[test]
    fn fun_call() {
        assert_compiles_to!(
            "(fun foo (x) x) (foo 1)",
            [
                ByteCode::Jump(3),
                ByteCode::Load(0),
                ByteCode::Ret, // Load arg (pushing it to top of stack)
                ByteCode::Push(1),
                ByteCode::Push(1), // number of args
                ByteCode::Call(1)
            ]
        );
    }

    #[test]
    fn fun_call_multiple_args() {
        assert_compiles_to!(
            "(fun foo (x y) (+ x y)) (foo 1 2)",
            [
            ByteCode::Jump(5),
            ByteCode::Load(1),
            ByteCode::Load(0),
            ByteCode::Add,
            ByteCode::Ret, // Load arg (pushing it to top of stack)
            ByteCode::Push(1),
            ByteCode::Push(2),
            ByteCode::Push(2), // number of args
            ByteCode::Call(1)
            ]
        );
    }

    #[test]
    fn fun_call_rec() {
        assert_compiles_to!(
            "(fun foo (x) (foo (- x 1))) (foo 10)",
            [
            ByteCode::Jump(7),
            ByteCode::Load(0),
            ByteCode::Push(1),
            ByteCode::Sub,
            ByteCode::Push(1),
            ByteCode::Call(1),
            ByteCode::Ret, // Load arg (pushing it to top of stack)
            ByteCode::Push(10),
            ByteCode::Push(1), // number of args
            ByteCode::Call(1)
            ]
        );
    }
}
