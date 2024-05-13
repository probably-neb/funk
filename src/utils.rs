// rust 0 - ben 1
macro_rules! steal {
    ($ty:ty, $x:expr) => {
        unsafe {
            let mut_ptr: *mut $ty = $x as *const $ty as *mut $ty;
            &mut *mut_ptr
        }
    };
}

pub(crate) use steal;

macro_rules! utf8_str {
    ($bytes:expr) => {
        unsafe { std::str::from_utf8_unchecked($bytes)}
    }
}

pub(crate) use utf8_str;
