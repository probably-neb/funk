use crate::parser;

pub struct Ast {
    pub exprs: Vec<parser::Expr>,
    pub data: DataPool,
}

impl Ast {
    pub fn new(exprs: Vec<parser::Expr>, data: DataPool) -> Self {
        Self { exprs, data }
    }

    pub fn get_const<T: ReadFromBytes>(&self, i: DIndex) -> T {
        return self.data.get::<T>(i);
    }

    pub fn get_const_ref<'a, T: ReadFromBytesRef<'a>>(&'a self, i: DIndex) -> &'a T {
        return self.data.get_ref::<T>(i);
    }
}

type Pool = Vec<u8>;

pub type DIndex = usize;

pub struct DataPool {
    pool: Pool,
}

// TODO: implement hashing on write for deduplication
// -> either use resulting bytes as key or come up with
//    alternate hashing method
// TODO: determine whether to use separate pools for
//      strings and ints. i.e. have array of just u32's, and just string bytes
impl DataPool {
    pub fn new() -> Self {
        return Self { pool: vec![] };
    }

    pub fn put<T: WriteBytes + ?Sized>(&mut self, val: &T) -> DIndex {
        let i = self.pool.len();
        WriteBytes::write(val, &mut self.pool);
        return i;
    }

    pub fn get<T: ReadFromBytes>(&self, i: DIndex) -> T {
        return ReadFromBytes::read(&self.pool[i..]);
    }

    pub fn get_ref<'a, T: ReadFromBytesRef<'a> + ?Sized>(&'a self, i: DIndex) -> &'a T {
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

macro_rules! impl_read_write_for_num {
    ($t:ty) => {
        impl WriteBytes for $t {
            fn write(&self, out: &mut Pool) {
                let arr = self.to_be_bytes();
                const N_BYTES: usize = std::mem::size_of::<$t>();
                out.try_reserve(N_BYTES).expect("pool reserve failed");
                out.extend(&arr);
            }
        }
        impl ReadFromBytes for $t {
            fn read(bytes: &[u8]) -> Self {
                const N_BYTES: usize = std::mem::size_of::<$t>();
                use anyhow::Context;
                let arr: [u8; N_BYTES] = bytes[0..N_BYTES]
                    .try_into()
                    .with_context(|| format!("{} bytes for {}", N_BYTES, stringify!($t)))
                    .unwrap();
                return <$t>::from_be_bytes(arr);
            }
        }
    };
}

impl_read_write_for_num!(u64);
impl_read_write_for_num!(u32);

impl WriteBytes for &str {
    fn write(&self, out: &mut Pool) {
        let bytes = self.as_bytes();
        WriteBytes::write(bytes, out);
    }
}

impl<'a> ReadFromBytesRef<'a> for str {
    fn read(bytes: &'a [u8]) -> &'a str {
        let str_bytes: &[u8] = ReadFromBytesRef::read(bytes);
        // TODO: consider using unsafe from_utf8_unchecked to skip validation
        return std::str::from_utf8(str_bytes).expect("valid utf8");
    }
}

impl WriteBytes for [u8] {
    fn write(&self, out: &mut Pool) {
        let len = self.len() as u32;

        let out_len = 4 + len;
        out.try_reserve(out_len as usize)
            .expect("pool reserve failed");

        WriteBytes::write(&len, out);
        out.extend_from_slice(&self);
    }
}

impl<'a> ReadFromBytesRef<'a> for [u8] {
    fn read(bytes: &'a [u8]) -> &'a [u8] {
        let len: u32 = ReadFromBytes::read(&bytes[0..4]);
        return &bytes[4..4 + len as usize];
    }
}

#[cfg(test)]
mod data_pool_tests {
    use super::*;

    macro_rules! test_read_write {
        ($t:ty, $val:expr) => {
            let mut pool = DataPool::new();
            let i = pool.put::<$t>(&$val);
            let read_val = pool.get::<$t>(i);
            assert_eq!($val, read_val);
        };
        (ref $t:ty, $val:expr) => {
            let mut pool = DataPool::new();
            let i = pool.put(&$val);
            let read_val: &$t = pool.get_ref::<$t>(i);
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
