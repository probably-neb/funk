use anyhow::Result;
use phf::phf_map;

pub type Range = (usize, usize);

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Token {
    Ident(Range),
    Int(Range),
    Float(Range),
    String(Range),
    Char(usize),
    Lt,
    LtEq,
    Gt,
    GtEq,
    Eq,
    DblEq,
    Plus,
    Minus,
    Mul,
    Div,
    Mod,
    And,
    LParen,
    RParen,
    LSquirly,
    RSquirly,
    LBrace,
    RBrace,
    True,
    False,
    Eof,
    If,
    Else,
    Fun,
    Let,
    Return,
    Colon,
    Comma,
    Pipe,
    While,
    Print,
}

#[derive(Clone)]
pub struct Lexer<'a> {
    position: usize,
    read_position: usize,
    ch: u8,
    input: &'a [u8],
}

static KEYWORDS: phf::Map<&'static [u8], Token> = phf_map! {
    b"true" => Token::True,
    b"false" => Token::False,
    b"if" => Token::If,
    b"else" => Token::Else,
    b"fun" => Token::Fun,
    b"let" => Token::Let,
    b"return" => Token::Return,
    b"while" => Token::While,
    b"and" => Token::And,
    b"print" => Token::Print,
};

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut lex = Self {
            position: 0,
            read_position: 0,
            ch: 0,
            input: input.as_bytes(),
        };
        lex.step();

        return lex;
    }

    // TODO: result/option required?
    pub fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace();

        let tok = match self.ch {
            b'"' => Token::String(self.read_string()),
            b'\'' => Token::Char(self.read_char()),
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => return Ok(self.ident_or_builtin()),
            b'0'..=b'9' => return Ok(self.read_numeric()),
            _ if self.ch.is_ascii_graphic() => self.read_symbol(),
            0 => Token::Eof,
            ch => unreachable!("unrecognized char: {}", ch as char),
        };

        self.step();
        return Ok(tok);
    }

    fn step(&mut self) {
        self.ch = self.peek();

        self.position = self.read_position;
        self.read_position += 1;
    }

    fn step_while<F>(&mut self, f: F)
    where
        F: Fn(u8) -> bool,
    {
        while f(self.ch) {
            self.step();
        }
    }

    fn step_then_expect(&mut self, byte: u8) {
        self.step();
        assert_eq!(
            self.ch, byte,
            "unexpected char {} in input, expected {}",
            self.ch, self.ch
        );
    }

    fn expect_then_step(&mut self, byte: u8) {
        assert_eq!(
            self.ch, byte,
            "unexpected char {} in input, expected {}",
            self.ch, self.ch
        );
        self.step();
    }

    fn peek(&self) -> u8 {
        return if self.read_position >= self.input.len() {
            0
        } else {
            self.input[self.read_position]
        };
    }

    fn skip_whitespace(&mut self) {
        self.step_while(|ch| ch.is_ascii_whitespace());
    }

    fn ident_or_builtin(&mut self) -> Token {
        let range = self.read_ident();
        debug_assert_ne!(range.0, range.1);
        let slice = self.slice(&range);
        if let Some(kw) = KEYWORDS.get(slice) {
            return *kw;
        }
        return Token::Ident(range);
    }

    fn read_ident(&mut self) -> Range {
        let pos = self.position;

        let is_valid_ident_char = |ch: u8| ch.is_ascii_alphabetic() || ch == b'_';

        self.step_while(is_valid_ident_char);

        return (pos, self.position);
    }

    fn read_numeric(&mut self) -> Token {
        let pos = self.position;
        self.step_while(|ch| ch.is_ascii_digit());
        if self.ch == b'.' || self.ch == b'e' {
            self.step();
            return Token::Float(self.read_float(pos));
        }
        return Token::Int((pos, self.position));
    }

    fn read_float(&mut self, start: usize) -> Range {
        self.step_while(|ch| ch.is_ascii_digit());
        if self.ch == b'e' {
            self.step();
            if self.ch == b'-' || self.ch == b'+' {
                self.step();
            }
            self.step_while(|ch| ch.is_ascii_digit());
        }
        return (start, self.position);
    }

    fn read_string(&mut self) -> Range {
        let pos = self.position + 1;
        // eat `"`
        self.expect_then_step(b'"');
        // debug_assert_eq!(self.ch, b'"');
        // self.step();
        self.step_while(|ch| ch != b'"' && ch != 0);
        // reached end of input while in string
        debug_assert_ne!(self.ch, 0);
        return (pos, self.position);
    }

    fn read_char(&mut self) -> usize {
        let pos = self.position + 1;
        self.step();
        self.step_then_expect(b'\'');
        return pos;
    }

    fn read_symbol(&mut self) -> Token {
        macro_rules! if_peek {
            ($char:literal, $a:expr, $b:expr) => {
                match self.peek() {
                    $char => {
                        self.step();
                        $a
                    }
                    _ => $b,
                }
            };
        }
        match self.ch {
            b'<' => if_peek!(b'=', Token::LtEq, Token::Lt),
            b'>' => if_peek!(b'=', Token::GtEq, Token::Gt),
            b'=' => if_peek!(b'=', Token::DblEq, Token::Eq),
            b'|' => if_peek!(b'>', Token::Pipe, unreachable!()),
            b'-' => Token::Minus,
            b'(' => Token::LParen,
            b')' => Token::RParen,
            b'{' => Token::LSquirly,
            b'}' => Token::RSquirly,
            b'[' => Token::LBrace,
            b']' => Token::RBrace,
            b'+' => Token::Plus,
            b'*' => Token::Mul,
            b'/' => Token::Div,
            b':' => Token::Colon,
            b',' => Token::Comma,
            b'%' => Token::Mod,
            _ => unreachable!("unrecognized punct {}", self.ch as char),
        }
    }

    // TODO: don't take ptr to range here
    pub fn slice(&self, range: &Range) -> &[u8] {
        let (start, end) = *range;
        debug_assert!(end <= self.input.len());
        return &self.input[start..end];
    }

    pub fn as_str(&self, range: &Range) -> &str {
        let slice = self.slice(range);
        return std::str::from_utf8(slice).unwrap();
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    macro_rules! assert_matches {
        ($expr:expr, $pat:pat) => {
            assert!(
                matches!($expr, $pat),
                "expected {:?}, got {:?}",
                stringify!($pat),
                $expr
            )
        };
        ($expr:expr, $pat:pat, $message:literal) => {
            assert!(matches!($expr, $pat), $message)
        };
        ($expr:expr, $pat:pat => $body:expr) => {{
            assert_matches!($expr, $pat);
            match $expr {
                $pat => $body,
                _ => unreachable!("unreachable pattern"),
            }
        }};
    }

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
        assert_eq!(lex.next_token().unwrap(), Token::Ident((0, 3)));
        assert_eq!(lex.next_token().unwrap(), Token::Plus);
        assert_eq!(lex.next_token().unwrap(), Token::Ident((4, 7)));
    }

    #[test]
    fn string() {
        let contents = "\"foo\"";
        let mut lex = Lexer::new(contents);
        let tok = lex.next_token().unwrap();
        let range = match tok {
            Token::String(range) => range,
            _ => unreachable!("expected string, got {:?}", tok),
        };
        assert_eq!(lex.as_str(&range), "foo");
    }

    #[test]
    fn char() {
        let contents = "'a'";
        let mut lex = Lexer::new(contents);
        let tok = lex.next_token().unwrap();
        assert_eq!(tok, Token::Char(1));
    }

    #[test]
    fn float() {
        let contents = "10.0";
        let mut lex = Lexer::new(contents);
        let tok = lex.read_numeric();
        let range = match tok {
            Token::Float(range) => range,
            _ => unreachable!("expected float, got {:?}", tok),
        };
        assert_eq!(lex.as_str(&range), "10.0");
    }

    #[test]
    fn bools() {
        let contents = "true false";
        let mut lex = Lexer::new(contents);
        assert_eq!(lex.next_token().unwrap(), Token::True);
        assert_eq!(lex.next_token().unwrap(), Token::False);
    }

    #[test]
    fn brackets() {
        let contents = "[](){}";
        let mut lex = Lexer::new(contents);
        assert_eq!(lex.next_token().unwrap(), Token::LBrace);
        assert_eq!(lex.next_token().unwrap(), Token::RBrace);
        assert_eq!(lex.next_token().unwrap(), Token::LParen);
        assert_eq!(lex.next_token().unwrap(), Token::RParen);
        assert_eq!(lex.next_token().unwrap(), Token::LSquirly);
        assert_eq!(lex.next_token().unwrap(), Token::RSquirly);
    }

    #[test]
    fn eq_dbl_eq() {
        let contents = "== =";
        let mut lex = Lexer::new(contents);
        assert_eq!(lex.next_token().unwrap(), Token::DblEq);
        assert_eq!(lex.next_token().unwrap(), Token::Eq);
    }

    #[test]
    fn lteq() {
        let contents = "<=";
        let mut lex = Lexer::new(contents);
        assert_eq!(lex.next_token().unwrap(), Token::LtEq);
    }

    #[test]
    fn gteq() {
        let contents = ">=";
        let mut lex = Lexer::new(contents);
        assert_eq!(lex.next_token().unwrap(), Token::GtEq);
    }

    #[test]
    fn float_scientific() {
        let contents = "1.0e10";
        let mut lex = Lexer::new(contents);
        let tok = lex.read_numeric();
        let range = match tok {
            Token::Float(range) => range,
            _ => unreachable!("expected float, got {:?}", tok),
        };
        assert_eq!(lex.as_str(&range), "1.0e10");
    }

    #[test]
    fn let_builtin() {
        let contents = "let";
        let mut lex = Lexer::new(contents);
        assert_eq!(lex.next_token().unwrap(), Token::Let);
    }

    #[test]
    fn type_annotations() {
        let contents = "fun foo(a int, b int) int { return  (+ a b)}";
        let mut lex = Lexer::new(contents);
        assert_eq!(lex.next_token().unwrap(), Token::LParen);
        assert_eq!(lex.next_token().unwrap(), Token::Fun);
        assert_eq!(lex.next_token().unwrap(), Token::LParen);

        assert_matches!(lex.next_token().unwrap(), Token::Ident(_));
        assert_eq!(lex.next_token().unwrap(), Token::Colon);
        assert_matches!(lex.next_token().unwrap(), Token::Ident(_));

        assert_matches!(lex.next_token().unwrap(), Token::Ident(_));
        assert_eq!(lex.next_token().unwrap(), Token::Colon);
        assert_matches!(lex.next_token().unwrap(), Token::Ident(_));

        assert_eq!(lex.next_token().unwrap(), Token::RParen);
        assert_eq!(lex.next_token().unwrap(), Token::Colon);
        assert_matches!(lex.next_token().unwrap(), Token::Ident(_));
    }
}
