use crate::lexer;
use crate::parser;

pub struct Ast<'a> {
    pub exprs: Vec<parser::Expr>,
    pub tokens: Vec<lexer::Token>,
    src: &'a [u8],
}

impl<'a> Ast<'a> {
    pub fn new(exprs: Vec<parser::Expr>, tokens: Vec<lexer::Token>, src: &'a [u8]) -> Self {
        Self {
            exprs,
            tokens,
            src,
        }
    }

    fn slice(&self, range: (usize, usize)) -> &'a [u8] {
        let start = range.0;
        let end = range.1.min(self.src.len());
        return &self.src[start..end];
    }
    pub fn token_slice(&self, tok_i: usize) -> &'a str {
        let tok = self.tokens[tok_i];
        let range =  match tok {
            lexer::Token::Ident(range) => range,
            lexer::Token::Int(range) => range,
            lexer::Token::Float(range) => range,
            lexer::Token::String(range) => range,
            _ => unreachable!("tried to get range of non-range token: {:?}", stringify!(tok))
        };
        let slice = self.slice(range);
        return std::str::from_utf8(slice).unwrap();
    }
}

type Pool = Vec<u8>;

pub struct DataPool {
    pool: Pool
}

impl DataPool {
    pub fn new() -> Self {
        return Self {
            pool: vec![]
        }
    }

    pub fn write<T: WriteBytes>(&mut self, val: &T) -> usize {
        let i = self.pool.len();
        WriteBytes::write(val, &mut self.pool);
        return i;
    }

    pub fn read<T: ReadFromBytes>(&mut self, i: usize) -> T {
        return ReadFromBytes::read(&self.pool[i..]);
    }

    pub fn read_ref<'a, T: ReadFromBytesRef<'a> + ?Sized>(&'a mut self, i: usize) -> &'a T {
        return ReadFromBytesRef::read(&self.pool[i..]);
    }
}

pub trait WriteBytes {
    fn write(&self, out: &mut Pool);
}

pub trait ReadFromBytes {
    fn read(bytes: &[u8]) -> Self;
}

pub trait ReadFromBytesRef<'a> {
    fn read(bytes: &'a [u8]) -> &'a Self;
}

impl WriteBytes for u32 {
    fn write(&self, out: &mut Pool) {
        let arr = self.to_be_bytes();
        out.try_reserve(4).expect("pool reserve failed");
        out.extend(&arr);
    }
}

impl ReadFromBytes for u32 {
    fn read(bytes: &[u8]) -> Self {
        let arr: [u8; 4] = bytes[0..4].try_into().expect("4 bytes for u32");
        return u32::from_be_bytes(arr);
    }
}

impl WriteBytes for &str {
    fn write(&self, out: &mut Pool) {
        let len = self.len() as u32;

        let out_len = 4 + len;
        out.try_reserve(out_len as usize).expect("pool reserve failed");

        WriteBytes::write(&len, out);
        let bytes = self.as_bytes();
        out.extend_from_slice(bytes);
    }
}

impl<'a> ReadFromBytesRef<'a> for str {
    fn read(bytes: &'a [u8]) -> &'a str {
        let len: u32 = ReadFromBytes::read(&bytes[0..4]);
        let str_bytes = &bytes[4..4 + len as usize];
        return std::str::from_utf8(str_bytes).expect("valid utf8");
    }
}

#[cfg(test)]
mod data_pool_tests {
    use super::*;

    macro_rules! test_read_write {
        ($t:ty, $val:expr) => {
            let mut pool = DataPool::new();
            let i = pool.write(&$val);
            let read_val = pool.read::<$t>(i);
            assert_eq!($val, read_val);
        };
        (ref $t:ty, $val:expr) => {
            let mut pool = DataPool::new();
            let i = pool.write(&$val);
            let read_val: &$t = pool.read_ref::<$t>(i);
            assert_eq!($val, read_val);
        };
    }

    #[test]
    fn u32() {
        test_read_write!(u32, 0x12345678);
    }

    #[test]
    fn str() {
        test_read_write!(ref str, "hello world");
    }
}
