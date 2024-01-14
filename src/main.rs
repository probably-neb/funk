mod lexer;

use anyhow::Result;

use lexer::{Lexer, Token};

fn main() -> Result<()> {
    let file = std::env::args().nth(1).expect("no file given");
    let contents = std::fs::read_to_string(file)?;
    let mut lex = Lexer::new(&contents);
    loop {
        let tok = lex.next_token()?;
        println!("{:?}", lex.repr(&tok));
        if tok == Token::Eof {
            break;
        }
    }
    Ok(())
}

