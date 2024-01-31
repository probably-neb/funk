use anyhow::{Result, Context};

use crate::compiler::ByteCode;

type Stack = Vec<u64>;

pub fn run(bytecode: &[ByteCode]) -> Result<Stack> {
    let mut stack = Vec::new();
    for bc in bytecode.into_iter() {
        match *bc {
            ByteCode::Push(i) => {
                stack.push(i);
            },
            ByteCode::Add => {
                let rhs = stack.pop().context("stack underflow")?;
                let lhs = stack.pop().context("stack underflow")?;
                stack.push(lhs + rhs);
            }
        }
    }
    Ok(stack)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn add() {
        let bytecode = vec![
            ByteCode::Push(1),
            ByteCode::Push(2),
            ByteCode::Add,
        ];
        let stack = run(&bytecode).unwrap();
        assert_eq!(vec![3], stack);
    }
}
