use anyhow::{Context, Result};

use crate::fir::{self, DIndex, XIndex, FIR};
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
            Binop::Add => ByteCode::Add,
            Binop::Sub => ByteCode::Sub,
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

pub struct Assembler {
    fir: FIR,
    bytecode: Chunk,
    cursor: usize,
    fun_map: FunMap,
    cur_fun_i: Option<usize>,
    scope: ScopeStack,
}

impl Assembler {
    pub fn assemble(fir: FIR) -> Result<Chunk> {
        let mut assembler = Assembler::new(fir);
        assembler._assemble();
        Ok(assembler.bytecode)
    }
    pub fn new(fir: FIR) -> Self {
        Self {
            fir,
            bytecode: Chunk { ops: Vec::new() },
            cursor: 0,
            cur_fun_i: None,
            fun_map: FunMap::new(),
            scope: ScopeStack::new(),
        }
    }

    fn next_op(&mut self) -> Option<fir::Op> {
        let op = self.fir.ops.get(self.cursor).copied();
        self.cursor += 1;
        op
    }

    pub fn _assemble(&mut self) {
        while let Some(op) = self.next_op() {
            self._assemble_op(op);
        }
    }

    fn _assemble_op(&mut self, op: fir::Op) {
        use fir::Op::*;
        match op {
            Load(ref_) => {
                self.assemble_ref(ref_);
                self.bind();
            },
            Add(lhs, rhs)
            | Sub(lhs, rhs)
            | Mul(lhs, rhs)
            | Div(lhs, rhs)
            | GtEq(lhs, rhs)
            | LtEq(lhs, rhs)
            | Lt(lhs, rhs)
            | Gt(lhs, rhs)
            | Eq(lhs, rhs) => {
                match lhs {
                    fir::Ref::Const(di) => {
                        self.inline_int(di);
                        self.assemble_ref(rhs);
                        self.scope.skip::<2>();
                    }
                    fir::Ref::Inst(i) => match self.scope.get(i).expect("not found") {
                        0 => {
                            self.assemble_ref(rhs);
                            self.scope.skip::<1>();
                        }
                        1 if references_top_of_stack(rhs, &self.scope) => {}
                        lhs_i => {
                            self.emit(ByteCode::Load(lhs_i));
                            self.assemble_ref(rhs);
                            self.scope.skip::<2>();
                        }
                    },
                }
                self.scope.pop(2);
                let bc_op = bc_binop_from_fir_binop(op).unwrap();
                self.emit_bind(bc_op);
            }
            FunDef { .. } => {
                let start = self.init_jmp();
                let fun_start_i = self.current_offset();
                self.cur_fun_i = Some(fun_start_i);
                self.fun_map.add(self.cursor - 1, fun_start_i as u32);

                self.scope.start_new();

                let mut op = self.next_op().expect("no fun body");

                while matches!(op, FunArg) {
                    self.bind();
                    op = self.next_op().expect("no fun body");
                }
                self.scope.skip::<2>();

                while !matches!(op, FunEnd) {
                    self._assemble_op(op);
                    op = self.next_op().expect("no fun body");
                }

                self.end_jmp(start);
                self.scope.end();
                self.cur_fun_i = None;
            }
            Ret(ref_) => {
                self.assemble_ref(ref_);
                self.emit(ByteCode::Ret);
            }
            FunCall { fun, args } => {
                let fun_offset = self.fun_map.get_offset(fun as usize).expect("fun exists");
                let bc = &mut self.bytecode;
                let crate::ast::ExtraFunArgs { args } = self.fir.extra.get(args);
                let num_args = args.len();
                for (arg_i, &arg) in args.iter().enumerate() {
                    let expected_i = num_args - arg_i - 1;
                    if !self.scope.is_at_offset(arg, expected_i) {
                        Self::emit_mut(bc, ByteCode::Load(arg));
                        self.scope.skip::<1>();
                    }
                }
                self.emit(ByteCode::Push(num_args as u64));
                self.emit_bind(ByteCode::Call(fun_offset));
            }
            Alloc => {
                self.bind();
            }
            Store(dest, src) => {
                // FIXME: don't store refs in Store
                let fir::Ref::Inst(dest) = dest else {
                    unreachable!("store dest not inst {:?}", dest);
                };
                self.assemble_ref(src);
                self.emit(ByteCode::Store(dest));
                self.scope.pop(1);
            }
            Branch { cond, t, f} => {
                self.assemble_ref(cond);
                let jmp_t = self.init_jmp();
                let jmp_f = self.init_jmp();
                // FIXME: Create separate type for Inst refs to avoid this
                let fir::Ref::Inst(t_i) = t else {
                    unreachable!("branch t not inst {:?}", t);
                };
                let fir::Ref::Inst(f_i) = f else {
                    unreachable!("branch f not inst {:?}", f);
                };
                self.end_jmp_if_zero(jmp_t);
                let end_t_op = self.assemble_basic_block(t_i);
                self.end_jmp_if_zero(jmp_f);
                let end_f_op = self.assemble_basic_block(f_i);
                self._assemble_op(end_t_op);
                self._assemble_op(end_f_op);

            }
            Jump(i) => {
                let jmp = self.init_jmp();
                let fir::Ref::Inst(jmp_to_i) = i else {
                    unreachable!("jump not inst {:?}", i);
                };
                self.end_jmp(jmp);
                let end_jmp_to_i = self.assemble_basic_block(jmp_to_i);

            }
            Phi {
                a: (a_from, a_res),
                b: (b_from, b_res),
            } => {
                
            }
            _ => unimplemented!("op: {:?} i: {}", op, self.cursor - 1),
        }
    }

    fn insert_phi_copies(&mut self, from_bb: u32, to_bb: u32) {
        let fir = &self.fir;
        let bb = slice_basic_block(fir, to_bb);
        for op in bb {
            let fir::Op::Phi {a, b} = op else {
                continue;
            };
            todo!("match on a,b checking if the from is from_bb. then insert necessary copies")
        }
    }

    fn assemble_basic_block(&mut self, start: u32) -> fir::Op {
        self.cursor = start as usize;
        let fir::Op::Label = self.fir.ops[self.cursor] else {
            unreachable!("not a basic block");
        };
        self.cursor += 1;
        while let Some(op) = self.next_op() {
            if op.is_ctrl_flow() {
                return op;
            }
            self._assemble_op(op);
        }
        unreachable!("no end of basic block");
    }

    fn assemble_ref(&mut self, r: fir::Ref) {
        match r {
            fir::Ref::Const(di) => {
                self.inline_int(di);
            }
            fir::Ref::Inst(i) => {
                let i = self.scope.get(i).with_context(|| i).expect("not found");
                if i == 0 {
                    return;
                }
                self.emit(ByteCode::Load(i));
            }
        }
    }

    fn assemble_load_inst(&mut self, i: u32) {
        let i = self.scope.get(i).expect("not found");
        if i == 0 {
            return;
        }
        self.emit(ByteCode::Load(i));
    }

    fn set(&mut self, i: usize, bc: ByteCode) {
        self.bytecode.ops[i] = bc;
    }

    fn emit_mut(chunk: &mut Chunk, bc: ByteCode) {
        chunk.ops.push(bc);
    }

    fn emit(&mut self, bc: ByteCode) {
        self.bytecode.ops.push(bc);
    }

    fn emit_bind(&mut self, bc: ByteCode) {
        self.bytecode.ops.push(bc);
        self.bind();
    }

    fn bind(&mut self) {
        let inst = (self.cursor - 1 - self.cur_fun_i.unwrap_or(0)) as u32;
        self.scope.bind_local(inst);
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

    fn inline_int(&mut self, di: u32) {
        let val: u64 = self.fir.get_const(di);
        self.emit(ByteCode::Push(val));
    }

    pub fn bytecode(&self) -> &Chunk {
        &self.bytecode
    }
}

fn slice_basic_block<'s>(fir: &'s fir::FIR, start: u32) -> &'s [fir::Op] {
    let start = start as usize;
    let mut end = start;
    while !fir.ops[end].is_ctrl_flow() && end < fir.ops.len() {
        end += 1;
    }
    return &fir.ops[start..=end];
}


fn references_top_of_stack(r: fir::Ref, scope: &ScopeStack) -> bool {
    match r {
        fir::Ref::Const(_) => false,
        fir::Ref::Inst(i) => {
            let last = scope.stack_map.last().expect("scope stack not empty");
            *last == i
        }
    }
}

fn bc_binop_from_fir_binop(op: fir::Op) -> Option<ByteCode> {
    use fir::Op;
    use ByteCode::*;
    Some(match op {
        Op::Add(..) => Add,
        Op::Sub(..) => Sub,
        Op::Mul(..) => Mul,
        Op::Div(..) => Div,
        Op::GtEq(..) => GtEq,
        Op::LtEq(..) => LtEq,
        Op::Lt(..) => Lt,
        Op::Gt(..) => Gt,
        Op::Eq(..) => Eq,
        _ => return None,
    })
}

#[derive(Debug)]
struct ScopeStack {
    stack_map: Vec<u32>,
    starts: Vec<usize>,
    cur: usize,
}

impl ScopeStack {
    const NOT_BOUND: u32 = u32::MAX;

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
    fn bind_local(&mut self, inst: u32) -> u32 {
        let i = self.stack_map.len();
        self.stack_map.push(inst);
        return (i - self.cur) as u32;
    }

    fn get(&self, inst: u32) -> Option<u32> {
        debug_assert_ne!(inst, Self::NOT_BOUND);
        let pos = self.stack_map.iter().rev().position(|&n| n == inst);
        let Some(pos) = pos else {
            return None;
        };
        if pos < self.cur {
            unimplemented!("globals not implemented");
        }
        return Some((pos - self.cur) as u32);
    }
    fn pop(&mut self, n: usize) {
        debug_assert!(self.stack_map.len() - n >= self.cur);
        self.stack_map.truncate(self.stack_map.len() - n);
    }
    fn is_last(&self, i: u32) -> bool {
        let i = self.cur + i as usize;
        let len = self.stack_map.len();
        return i + 1 == len;
    }
    fn is_at_offset(&self, i: u32, offset: usize) -> bool {
        let vi = self.stack_map.len() - 1 - offset;
        let v = self.stack_map[vi];
        v == i
    }
}

struct FunMap {
    offsets: Vec<u32>,
    fir_locs: Vec<usize>,
}

impl FunMap {
    fn new() -> Self {
        Self {
            offsets: vec![],
            fir_locs: vec![],
        }
    }

    fn add(&mut self, fir_loc: usize, offset: u32) {
        self.fir_locs.push(fir_loc);
        self.offsets.push(offset);
    }

    fn get_offset(&self, loc: usize) -> Option<u32> {
        self.fir_locs
            .iter()
            .position(|&l| l == loc)
            .map(|i| self.offsets[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::ByteCode::*;
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

    fn assemble(contents: &str) -> Chunk {
        let parser = crate::parser::Parser::new(contents);
        let ast = parser.parse().expect("syntax error");
        let fir = crate::fir::FIRGen::generate(ast).expect("syntax error");
        println!("== FIR ==");
        print!("{}", fir.stringify());
        let bytecode = Assembler::assemble(fir).expect("compile error");
        println!("\n== Bytecode ==");
        for (i, bc) in bytecode.ops.iter().enumerate() {
            println!("{}: {:?}", i, bc);
        }
        return bytecode;
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
            let bytecode = assemble(contents);
            assert_bytecode_matches!(bytecode, [$($ops),*]);
        };
    }

    #[test]
    fn add() {
        assert_compiles_to!(
            "( + 1 2 )",
            [ByteCode::Push(1), ByteCode::Push(2), ByteCode::Add]
        );
    }

    #[test]
    fn nested_add() {
        assert_compiles_to!("( + 1 ( + 2 3 ) )", [Push(1), Push(2), Push(3), Add, Add]);
    }

    #[test]
    fn repeated_add() {
        assert_compiles_to!(
            "(+ 1 2) (+ 3 4)",
            [Push(1), Push(2), Add, Push(3), Push(4), Add]
        );
    }

    #[test]
    fn all_binops() {
        assert_compiles_to!(
            "(+ 1 2) (- 3 4) (* 5 6) (/ 7 8)",
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
        assert_compiles_to!(
            "(if 1 2 3)",
            [Push(1), JumpIfZero(3), Jump(5), Push(2), Jump(6), Push(3)]
        );
    }

    #[test]
    fn let_bind() {
        assert_compiles_to!("(let x 1)", [ByteCode::Push(1), ByteCode::Store(0)]);
    }

    #[test]
    fn let_bind_and_use() {
        assert_compiles_to!("(let x 1) (+ x 1)", [Push(1), Store(0), Push(1), Add]);
    }

    #[test]
    fn fun_def() {
        assert_compiles_to!(
            "(fun foo (x) x)",
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
    fn fun_def_multiple_args() {
        assert_compiles_to!("(fun foo (x y z) (+ x (+ y z)))", [
            Jump(_),
            Load(4),
            Load(4),
            Load(4),
            Add,
            Add,
            Ret
        ]);
    }

    #[test]
    fn fun_call() {
        assert_compiles_to!(
            "(fun foo (x) x) (foo 1)",
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
            "(fun foo (x y) (+ x y)) (foo 1 2)",
            [
                Jump(5),
                Load(3),
                Load(3),
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
            "(fun foo (x) (foo (- x 1))) (foo 10)",
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
                (fun countdown (n) (
                    if (> n 0)
                        (countdown (- n 1))
                        n
                ))
                (countdown 10)
            "#,
            [
                Jump(_),
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
