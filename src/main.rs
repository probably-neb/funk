mod lexer;
mod parser;
mod ast;
mod compiler;
mod rt;

use anyhow::Result;

fn main() -> Result<()> {
    let file = std::env::args().nth(1).expect("no file given");
    let contents = std::fs::read_to_string(file)?;
    let parser = parser::Parser::new(&contents);
    let ast = parser.parse()?;
    let mut compiler = compiler::Compiler::new(ast);
    compiler.compile();
    let bytecode = compiler.bytecode();
    dbg!(bytecode);
    let stack = rt::run(bytecode)?;
    dbg!(stack);
    Ok(())
}
