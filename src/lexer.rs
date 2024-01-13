use anyhow::Result;
type Range = (usize, usize);

#[derive(Debug, PartialEq)]
pub enum Token {
    Ident(Range),
    Punct(usize),
    Int(Range),
    Eof,
}

pub struct Lexer<'a> {
    position: usize,
    read_position: usize,
    ch: u8,
    input: &'a [u8],
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut lex = Self {
            position: 0,
            read_position: 0,
            ch: 0,
            input: input.as_bytes(),
        };
        lex.read_char();

        return lex;
    }

    pub fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace();

        let tok = match self.ch {
            _ if (self.ch as char).is_ascii_punctuation() => Token::Punct(self.position),
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let range = self.read_ident();
                return Ok(Token::Ident(range));
            },
            b'0'..=b'9' => return Ok(Token::Int(self.read_int())),
            0 => Token::Eof,
            ch => unreachable!("unrecognized char: {}", ch as char)
        };

        self.read_char();
        return Ok(tok);
    }

    fn peek(&self) -> u8 {
        if self.read_position >= self.input.len() {
            return 0;
        } else {
            return self.input[self.read_position];
        }
    }

    fn read_char(&mut self) {
        if self.read_position >= self.input.len() {
            self.ch = 0;
        } else {
            self.ch = self.input[self.read_position];
        }

        self.position = self.read_position;
        self.read_position += 1;
    }

    fn skip_whitespace(&mut self) {
        while self.ch.is_ascii_whitespace() {
            self.read_char();
        }
    }

    fn read_ident(&mut self) -> Range {
        let pos = self.position;
        while self.ch.is_ascii_alphabetic() || self.ch == b'_' {
            self.read_char();
        }

        return (pos, self.position);
    }

    fn read_int(&mut self) -> Range {
        let pos = self.position;
        while self.ch.is_ascii_digit() {
            self.read_char();
        }

        return (pos, self.position);
    }
    fn str_from_range(&self, range: &Range) -> &str {
        return std::str::from_utf8(&self.input[range.0..range.1]).unwrap();
    }
    pub fn print_token(&self, token: &Token) {
        match token {
            Token::Ident(range) => {
                println!("Ident({})", self.str_from_range(range));
            },
            Token::Punct(pos) => {
                println!("Punct({})", self.input[*pos] as char);
            },
            Token::Int(range) => {
                println!("Int({})", self.str_from_range(range));
            },
            Token::Eof => {
                println!("Eof");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int() {
        let contents = "10";
        let mut lex = Lexer::new(contents);
        assert!(lex.read_int() == (0, 2));
    }

    #[test]
    fn ident_with_underscore() {
        let contents = "foo_bar";
        let mut lex = Lexer::new(contents);
        assert!(lex.read_ident() == (0, contents.len()));
    }

    #[test]
    fn ident() {
        let contents = "foobar";
        let mut lex = Lexer::new(contents);
        assert!(lex.read_ident() == (0, contents.len()));
    }

    #[test]
    fn successive_ident() {
        let contents = "foobar baz";
        let mut lex = Lexer::new(contents);
        assert!(lex.read_ident() == (0, 6));
        lex.skip_whitespace();
        assert!(lex.read_ident() == (7, contents.len()));
    }

    #[test]
    fn skip_ws_then_read() {
        let contents = "  foobar";
        let mut lex = Lexer::new(contents);
        lex.skip_whitespace();
        assert!(lex.read_ident() == (2, contents.len()));
    }

    #[test]
    fn punct_sep_ident() {
        let contents = "foo+bar";
        let mut lex = Lexer::new(contents);
        assert!(lex.read_ident() == (0, 3));
        assert!(lex.next_token().unwrap() == Token::Punct(3));
        let range = lex.read_ident();
        assert!(lex.str_from_range(&range) == "bar");
    }
}
