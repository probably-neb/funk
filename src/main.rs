mod lexer;
mod parser;
mod ast;
mod compiler;
mod rt;
mod utils;
// mod fir;
// mod lower;

use anyhow::Result;

fn main() -> Result<()> {
    let file = std::env::args().nth(1).expect("no file given");
    let contents = std::fs::read_to_string(file)?;
    let parser = parser::Parser::new(&contents);
    let mut ast = parser.parse()?;
    ast::typecheck::typecheck(&mut ast)?;
    let mut compiler = compiler::Compiler::new(ast);
    compiler.compile();
    let bytecode = compiler.bytecode();
    dbg!(bytecode);
    let stack = rt::run(bytecode)?;
    dbg!(stack);
    Ok(())
}
