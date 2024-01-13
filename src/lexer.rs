use anyhow::Result;
type Range = (usize, usize);

#[derive(Debug, PartialEq)]
pub enum Token {
    Ident(Range),
    Punct(usize),
    Int(Range),
    Float(Range),
    String(Range),
    True,
    False,
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
            b'"' => Token::String(self.read_string()),
            _ if self.ch.is_ascii_punctuation() => Token::Punct(self.position),
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let range = self.read_ident();
                return Ok(self.ident_or_builtin(&range));
            },
            b'0'..=b'9' => return Ok(self.read_numeric()),
            0 => Token::Eof,
            ch => unreachable!("unrecognized char: {}", ch as char)
        };

        self.read_char();
        return Ok(tok);
    }

    fn ident_or_builtin(&self, range: &Range) -> Token {
        return match self.slice(&range) {
            b"true" => Token::True,
            b"false" => Token::False,
            _ => Token::Ident(*range)
        };
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

    fn read_numeric(&mut self) -> Token {
        let pos = self.position;
        while self.ch.is_ascii_digit() {
            self.read_char();
        }
        // TODO: check for floats in scientific notation
        if self.ch == b'.' {
            self.read_char();
            return Token::Float(self.read_float(pos));
        }

        return Token::Int((pos, self.position));
    }

    fn read_float(&mut self, start: usize) -> Range {
        while self.ch.is_ascii_digit() {
            self.read_char();
        }
        return (start, self.position)
    }

    fn read_string(&mut self) -> Range {
        let pos = self.position + 1;
        while self.peek() != b'"' && self.ch != 0 {
            self.read_char();
        }
        self.read_char();
        return (pos, self.position);
    }

    fn slice(&self, range: &Range) -> &[u8] {
        let start = range.0;
        let end = range.1.min(self.input.len());
        return &self.input[start..end];
    }

    fn str_from_range(&self, range: &Range) -> &str {
        let slice = self.slice(range);
        return std::str::from_utf8(slice).unwrap();
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
            Token::Float(range) => {
                println!("Float({})", self.str_from_range(range));
            },
            Token::String(range) => {
                println!("String(\"{}\")", self.str_from_range(range));
            },
            Token::True => {
                println!("True");
            },
            Token::False => {
                println!("False");
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
        assert!(lex.read_numeric() == Token::Int((0, 2)));
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

    #[test]
    fn string() {
        let contents = "\"foo\"";
        let mut lex = Lexer::new(contents);
        let tok = lex.next_token().unwrap();
        let range = match tok {
            Token::String(range) => range,
            _ => unreachable!("expected string, got {:?}", tok)
        };
        assert_eq!(lex.str_from_range(&range), "foo");
    }

    #[test]
    fn float() {
        let contents = "10.0";
        let mut lex = Lexer::new(contents);
        let tok = lex.read_numeric();
        let range = match tok {
            Token::Float(range) => range,
            _ => unreachable!("expected float, got {:?}", tok)
        };
        assert_eq!(lex.str_from_range(&range), "10.0");
    }

    #[test]
    fn bools() {
        let contents = "true false";
        let mut lex = Lexer::new(contents);
        assert_eq!(lex.next_token().unwrap(), Token::True);
        assert_eq!(lex.next_token().unwrap(), Token::False);
    }
}
