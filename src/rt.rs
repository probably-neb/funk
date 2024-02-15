use anyhow::{Context, Result};

use crate::compiler::{ByteCode, Chunk};

type Stack = Vec<i64>;

pub fn run(chunk: &Chunk) -> Result<Stack> {
    let mut rt = Runtime::new(chunk);
    return rt.run();
}

struct Runtime<'chunk> {
    chunk: &'chunk Chunk,
    stack: Stack,
    pc: usize,
    vars: Vec<i64>,
    call_stack: Vec<CallFrame>,
    current_frame_offset: usize,
}

impl<'chunk> Runtime<'chunk> {
    fn new(chunk: &'chunk Chunk) -> Self {
        Self {
            chunk,
            stack: Vec::new(),
            pc: 0,
            vars: Vec::new(),
            call_stack: vec![],
            current_frame_offset: 0,
        }
    }

    fn next_instr(&mut self) -> Result<Option<ByteCode>> {
        if self.pc >= self.chunk.ops.len() {
            if self.pc > self.chunk.ops.len() {
                return Err(anyhow::anyhow!("pc out of range"));
            }
            return Ok(None);
        }
        let instr = self.chunk.ops[self.pc];
        self.pc += 1;
        return Ok(Some(instr));
    }

    fn run(&mut self) -> Result<Stack> {
        while let Some(bc) = self.next_instr()? {
            match bc {
                ByteCode::Push(i) => {
                    self.stack.push(i.try_into()?);
                }
                ByteCode::Add
                | ByteCode::Sub
                | ByteCode::Mul
                | ByteCode::Div
                | ByteCode::Eq
                | ByteCode::Lt
                | ByteCode::LtEq
                | ByteCode::Gt
                | ByteCode::GtEq => {
                    self.exec_binop(bc)?;
                }
                ByteCode::Mov(ofs) => self.exec_mov(ofs)?,
                ByteCode::JumpIfZero(addr) => self.exec_jmpiz(addr)?,
                ByteCode::Jump(addr) => self.exec_jmp(addr)?,
                ByteCode::Store(idx) => self.exec_store_local(idx)?,
                ByteCode::Load(idx) => self.exec_load_local(idx)?,
                ByteCode::Call(idx) => self.exec_call(idx)?,
                ByteCode::Ret => self.exec_ret()?,
                ByteCode::Nop => unreachable!("nop encountered at runtime"),
            }
        }
        Ok(self.stack.to_owned())
    }

    fn exec_jmpiz(&mut self, addr: u32) -> Result<()> {
        let cond = self.stack.pop().context("stack underflow")?;
        if cond == 0 {
            self.pc = addr as usize;
        }
        Ok(())
    }

    fn exec_jmp(&mut self, addr: u32) -> Result<()> {
        self.pc = addr as usize;
        Ok(())
    }

    fn exec_binop(&mut self, bc: ByteCode) -> Result<()> {
        let rhs = self.stack.pop().context("stack underflow")?;
        let lhs = self.stack.pop().context("stack underflow")?;
        let res = match bc {
            ByteCode::Add => lhs + rhs,
            ByteCode::Sub => lhs - rhs,
            ByteCode::Mul => lhs * rhs,
            ByteCode::Div => lhs / rhs,
            // FIXME: have JMPT and JMPF, zero as true is confusing and
            // results in (1 - ...) being the implementation of equals
            ByteCode::Eq => 1 - ((lhs == rhs) as i64),
            ByteCode::LtEq => 1 - ((lhs <= rhs) as i64),
            _ => unimplemented!("op: {:?} not implemented", bc),
        };
        self.stack.push(res);
        Ok(())
    }

    fn ensure_var_cap(&mut self, idx: usize) {
        if idx >= self.vars.len() {
            // NOTE: assumes uninitialized vars won't be accessed
            self.vars.resize(idx + 1, 0);
        }
    }

    fn exec_store_local(&mut self, idx: u32) -> Result<()> {
        let val = self.stack.pop().context("stack underflow")?;
        let offset = self.current_frame_offset;
        let idx = idx as usize + offset;
        self.ensure_var_cap(idx);
        self.vars[idx] = val;
        Ok(())
    }

    fn exec_load_local(&mut self, idx: u32) -> Result<()> {
        let offset = self.current_frame_offset;
        let idx = idx as usize + offset;
        let val = self.vars[idx];
        self.stack.push(val);
        Ok(())
    }

    fn exec_mov(&mut self, ofs: u32) -> Result<()> {
        let ofs = ofs as usize;
        let idx = self.stack.len() - ofs;
        self.stack[idx..].rotate_left(1);
        Ok(())
    }

    fn exec_call(&mut self, idx: u32) -> Result<()> {
        let addr = idx as usize;
        let fp = self.vars.len();
        let n_args = self.stack.pop().context("stack underflow")?;
        for _ in 0..n_args {
            let val = self.stack.pop().context("stack underflow")?;
            self.vars.push(val);
        }
        let pc = self.pc;
        self.call_stack.push(CallFrame {
            pc,
            heap_offset: fp,
        });
        self.current_frame_offset = fp;
        self.pc = addr;
        Ok(())
    }

    fn exec_ret(&mut self) -> Result<()> {
        let frame = self.call_stack.pop().context("call stack underflow")?;
        self.pc = frame.pc;
        self.vars.truncate(frame.heap_offset);

        if let Some(frame) = self.call_stack.last() {
            self.current_frame_offset = frame.heap_offset;
        } else {
            self.current_frame_offset = 0;
        }
        Ok(())
    }
}

struct CallFrame {
    pc: usize,
    heap_offset: usize,
}

#[cfg(test)]
mod test {
    use super::*;

    fn run_src(contents: &str) -> Stack {
        let parser = crate::parser::Parser::new(&contents);
        let ast = parser.parse().expect("parse error");
        let mut compiler = crate::compiler::Compiler::new(ast);
        compiler.compile();
        let chunk = compiler.bytecode();
        dbg!(chunk.ops.iter().enumerate().collect::<Vec<_>>());
        run(&chunk).expect("runtime error")
    }

    #[test]
    fn add() {
        let bytecode = vec![ByteCode::Push(1), ByteCode::Push(2), ByteCode::Add];
        let chunk = Chunk { ops: bytecode };
        let stack = run(&chunk).unwrap();
        assert_eq!(vec![3], stack);
        assert_eq!(stack.len(), 1);
    }

    #[test]
    fn jump_if_zero_not_taken() {
        let stack = run_src("(if (- 1 2) 3 4)");
        assert_eq!(stack, vec![4]);
    }

    #[test]
    fn eq() {
        let stack = run_src("(if (= 1 2) 3 4)");
        assert_eq!(stack, vec![4]);
    }

    #[test]
    fn lteq() {
        // lt
        let stack = run_src("(if (<= 1 2) 3 4)");
        assert_eq!(stack, vec![3]);
        // eq
        let stack = run_src("(if (<= 2 2) 3 4)");
        assert_eq!(stack, vec![3]);
    }

    #[test]
    fn use_var() {
        let stack = run_src("(let x 1) (+ x 2)");
        assert_eq!(stack, vec![3]);
    }

    #[test]
    fn fib() {
        let contents = r#"
            (fun fib (n) (
                if (<= n 1)
                    n
                    (+
                              (fib (- n 1))
                              (fib (- n 2))
                              )
                )
           )

            (fib 10)
        "#;
        let stack = run_src(contents);
        // fib(10)
        assert_eq!(stack, vec![55]);
    }

    #[test]
    fn mov() {
        let chunk = Chunk {
            ops: vec![ByteCode::Mov(2)],
        };
        let mut rt = Runtime::new(&chunk);
        rt.stack = vec![1, 2, 3];
        let stack = rt.run().unwrap();
        assert_eq!(stack, vec![1, 3, 2]);
    }
}
