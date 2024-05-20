pub mod stringify;
pub mod typecheck;

use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher}, num::NonZeroUsize,
};

use crate::parser::{self, XIndex};
// TODO: move XIndex, Expr, Binop here
// TODO: rename Expr to Node
pub use parser::{EIndex, Binop};

pub struct Ast {
    pub exprs: Vec<Expr>,
    pub data: DataPool,
    pub extra: Extra,
    pub types: Vec<Type>,
    /// References to other nodes in the AST
    /// Generated during name resolution in typecheck
    pub refs: Vec<RefIdent>,
}

impl Ast {
    pub fn new(exprs: Vec<parser::Expr>, data: DataPool, extra: Extra, types: Vec<Type>) -> Self {
        let num_exprs = exprs.len();
        Self { exprs, data, extra, types, refs: vec![None; num_exprs]}
    }

    pub fn get_const<T: ReadFromBytes>(&self, i: DIndex) -> T {
        return self.data.get::<T>(i);
    }

    #[allow(unused)]
    fn get_ident<'s>(&'s self, i: usize) -> &'s str {
        return self.data.get_ref::<str>(i);
    }

    #[allow(unused)]
    pub fn get_const_ref<'a, T: ReadFromBytesRef<'a>>(&'a self, i: DIndex) -> &'a T {
        return self.data.get_ref::<T>(i);
    }

    pub fn get_num_args(&self, first_arg: parser::EIndex) -> u32 {
        return self.extra[first_arg];
    }

    pub fn args_range(&self, args_i: parser::EIndex) -> std::ops::Range<parser::EIndex> {
        let len = self.get_num_args(args_i);
        let start = args_i + 1;
        let end = start + len as usize;
        return start..end;
    }

    pub fn fun_args_slice<'s>(&'s self, i: XIndex) -> &'s [u32] {
        return self.extra.get::<ExtraFunArgs>(i).args;
    }

    pub fn fun_arg_names_iter<'s>(&'s self, i: XIndex) -> impl Iterator<Item = &'s str> {
        return self.fun_args_slice(i).iter().map(|a| self.data.get_ref::<str>(*a as usize));
    }

    pub fn print(&self) {
        stringify::print_tree(self);
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Expr {
    Nop,
    Int(DIndex),
    Ident(DIndex),
    String(DIndex),
    If {
        cond: EIndex,
        branch_then: XIndex,
        branch_else: XIndex,
    },
    Binop {
        op: Binop,
        lhs: EIndex,
        rhs: EIndex,
    },
    FunDef {
        name: DIndex,
        args: XIndex,
        body: XIndex,
    },
    FunArg,
    FunCall {
        name: DIndex,
        args: XIndex,
    },
    Bind {
        name: DIndex,
        value: EIndex,
    },
    Return {
        value: Option<NonZeroUsize>
    },
    Bool(bool),
    While {
        cond: EIndex,
        body: XIndex,
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Unknown,
    Void,
    UInt64,
    String,
    Bool,
}

pub type RefIdent = Option<EIndex>;

// Wraps a HashMap<u64, DIndex> to use byte slices as keys
// by hashing the byte slice and using the resulting u64
// as the key
// prevents needing lifetimes everywhere
#[derive(Clone)]
struct ByteHashMap {
    map: HashMap<u64, DIndex>,
}

impl ByteHashMap {
    fn new() -> Self {
        return Self {
            map: HashMap::new(),
        };
    }
    fn hash(&self, key: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        // PERF: figure out how to avoid hashing things twice
        key.hash(&mut hasher);
        let hash = hasher.finish();
        return hash;
    }
    fn insert(&mut self, key: &[u8], val: DIndex) {
        let hash = self.hash(key);
        self.map.insert(hash, val);
    }
    fn get(&self, key: &[u8]) -> Option<DIndex> {
        let hash = self.hash(key);
        return self.map.get(&hash).copied();
    }
}

type Pool = Vec<u8>;

pub type DIndex = usize;

#[derive(Clone)]
pub struct DataPool {
    pool: Pool,
    tmp: Vec<u8>,
    map: ByteHashMap,
}

// TODO: implement hashing on write for deduplication
// -> either use resulting bytes as key or come up with
//    alternate hashing method
// TODO: determine whether to use separate pools for
//      strings and ints. i.e. have array of just u32's, and just string bytes
impl DataPool {
    pub fn new() -> Self {
        return Self {
            pool: vec![],
            map: ByteHashMap::new(),
            tmp: vec![],
        };
    }

    pub fn put<T: WriteBytes + ?Sized + Hash>(&mut self, val: &T) -> DIndex {
        // WARN: this puts the len of slices in `tmp` as well making it harder
        // making the write longer and harder to get the ID of a string
        // has the consequence we need write to a tmp buffer when getting the ID
        // where the tmp must be allocated so a mutable reference to self is not
        // required (could also use unsaafe)
        WriteBytes::write(val, &mut self.tmp);
        if let Some(i) = self.map.get(&self.tmp) {
            self.tmp.clear();
            return i;
        }
        let i = self.pool.len();
        self.map.insert(&self.tmp, i);
        self.pool.append(&mut self.tmp);
        return i;
    }

    pub fn get_ident_id(&self, val: &str) -> Option<DIndex> {
        let mut tmp = Vec::with_capacity(val.len() + 4);
        WriteBytes::write(&val, &mut tmp);
        return self.map.get(&tmp);
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

type ExtraData = u32;

#[derive(Clone)]
pub struct Extra {
    pub data: Vec<ExtraData>,
}

impl Extra {
    const RESERVE_VALUE: ExtraData = ExtraData::MAX;

    pub fn new() -> Self {
        return Self { data: vec![] };
    }

    pub fn get<'e, T: FromExtra<'e>>(&'e self, i: usize) -> T {
        return T::from_extra(self, i);
    }

    pub fn reserve(&mut self) -> usize {
        return self.append(Self::RESERVE_VALUE);
    }

    pub fn append(&mut self, val: ExtraData) -> usize {
        let i = self.data.len();
        self.data.push(val);
        return i;
    }

    pub fn append_u32(&mut self, val: ExtraData) -> u32 {
        let i = self.data.len();
        self.data.push(val);
        return i as u32;
    }

    pub fn concat(&mut self, data: &[ExtraData]) -> usize {
        let len = data.len() as u32;
        self.data.reserve_exact(data.len() + 1);
        let i = self.append(len);
        self.data.extend_from_slice(data);
        return i;
    }

    pub fn slice(&self, i: usize) -> &[ExtraData] {
        let len = self.data[i] as usize;
        let start = i + 1;
        if start == self.data.len() {
            return &[];
        }
        let end = start + len;
        return &self.data[start..end];
    }

    pub fn len_of(&self, i: usize) -> usize {
        return self.data[i] as usize;
    }

    /// Like `iter_of` but gives the index the item was found at as well
    pub fn indexed_iter_of<'a, T>(&'a self, i: usize, items: &'a [T]) -> impl Iterator<Item = (usize, &'a T)> + 'a {
        return self.iter(i).map(|i| {
            let i = i as usize;
            (i, &items[i])
        });
    }

    pub fn iter<'a>(&'a self, i: usize) -> impl Iterator<Item = ExtraData> + 'a {
        let len = self.data[i] as usize;
        let start = i + 1;
        let end = start + len;
        return (start..end).map(|i| self.data[i]);
    }

    pub fn iter_of<'a, T>(&'a self, i: usize, items: &'a [T]) -> impl Iterator<Item = &'a T> + 'a {
        return self.iter(i).map(|i| &items[i as usize]);
    }

    pub fn first_of(&self, i: usize) -> ExtraData {
        return self.data[i + 1];
    }

    pub fn last_of(&self, i: usize) -> ExtraData {
        return self.data[i + self.data[i] as usize];
    }

    pub fn fun_args_slice(&self, i: XIndex) -> &[u32] {
        return self.get::<ExtraFunArgs>(i).args;
    }

    pub fn fun_num_args(&self, i: XIndex) -> u32 {
        return self[i];
    }
}

impl std::ops::Index<usize> for Extra {
    type Output = ExtraData;
    fn index(&self, i: usize) -> &Self::Output {
        return &self.data[i];
    }
}

impl std::ops::IndexMut<usize> for Extra {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        return &mut self.data[i];
    }
}

impl std::ops::Index<u32> for Extra {
    type Output = ExtraData;
    fn index(&self, i: u32) -> &Self::Output {
        return &self.data[i as usize];
    }
}

impl std::ops::IndexMut<u32> for Extra {
    fn index_mut(&mut self, i: u32) -> &mut Self::Output {
        return &mut self.data[i as usize];
    }
}

pub trait FromExtra<'e> {
    fn from_extra(extra: &'e Extra, i: usize) -> Self;
}

pub struct ExtraFunArgs<'e> {
    pub args: &'e [ExtraData]
}

impl<'e> FromExtra<'e> for ExtraFunArgs<'e> {
    fn from_extra(extra: &'e Extra, i: usize) -> Self {
        let len = extra.data[i];
        let start = i + 1;
        let end = start + len as usize;
        let args = &extra.data[start..end];
        return Self { args };
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Range<T: TryFrom<u32> + Copy> {
    pub start: u32,
    pub end: u32,
    marker: std::marker::PhantomData<T>,
}

impl Range<usize> {
    pub fn new(start: usize, end: usize) -> Self {
        let start = start as u32;
        let end = end as u32;
        Range { start, end, marker: std::marker::PhantomData }
    }

    pub fn start_i(&self) -> usize {
        return self.start as usize;
    }

    pub fn end_i(&self) -> usize {
        return self.end as usize;
    }

    pub fn tup(&self) -> (u32, u32) {
        return (self.start, self.end);
    }
    pub fn tup_i(&self) -> (usize, usize) {
        return (self.start_i(), self.end_i());
    }

    pub fn range(&self) -> std::ops::Range<usize> {
        return self.start_i()..self.end_i();
    }

    pub fn slice<'a, T>(&self, data: &'a [T]) -> &'a [T] {
        return &data[self.start_i()..self.end_i()];
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    mod data_pool {

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

        #[test]
        fn interned() {
            let mut pool = DataPool::new();
            let i1 = pool.put(&"hello");
            let i2 = pool.put(&"hello");
            assert_eq!(i1, i2);
        }

        #[test]
        fn interned_diff() {
            let mut pool = DataPool::new();
            let i1 = pool.put(&"foo");
            let i2 = pool.put(&"bar");
            assert_ne!(i1, i2);
            assert_eq!("foo", pool.get_ref::<str>(i1));
            assert_eq!("bar", pool.get_ref::<str>(i2));
        }
    }
}
