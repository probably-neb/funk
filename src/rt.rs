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
}

impl<'chunk> Runtime<'chunk> {
    fn new(chunk: &'chunk Chunk) -> Self {
        Self {
            chunk,
            stack: Vec::new(),
            pc: 0,
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
                | ByteCode::Eq => {
                    self.exec_binop(bc)?;
                }
                ByteCode::JumpIfZero(addr) => self.exec_jmpiz(addr)?,
                ByteCode::Jump(addr) => self.exec_jmp(addr)?,
                _ => unimplemented!(),
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
            // results in (!= as i64) being the implementation of equals
            ByteCode::Eq => dbg!((lhs != rhs) as i64),
            _ => unimplemented!("op: {:?} not implemented", bc),
        };
        dbg!(bc, lhs, rhs, res);
        self.stack.push(res);
        Ok(())
    }
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
}
