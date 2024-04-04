# features list

- static type system
    - As much type inference as possible
- HOWTO: immutable declarations by default, mutable when declared mutable
    - opportunistic in-place mutation for immutable values
        - statically in compiler (see Roc! "alias analysis via Morphic solver")
        - if using RC, don't do copy if RC is 1!
- As fast as possible compiler
- Non-Blocking type errors (compiler places equivalent of Rust's `unreachable!()` before places a type error was found allowing code to be run)
    - Note: See Roc for implementation
- lsp, formatter, etc included with compiler
- ability to compile to binary?
    - makes bit-rot less likely
- **delightful** error messages (see Elm, Rust)


# References

- "Why static typing came back" talk by Richard Feldman


# let, mut, ref oh my!

## let

- value semantics
- **cannot** be rebound (or by extension mutated)
- always(1) copies when "moved" (bound to new ident, passed to fn)
- cannot be passed to `mut` fn params
[*]
| when passing `let` var to func copy not necessarily
| required as the var cannot be modified.
| however whether to copy would have to be determined
| based on size of val and how vals are stored in vm
| if it would provide a runtime improvement

## mut

- value semantics
- **can** be rebound (and by extension mutated)
- mutating field creates new copy(2) with change
- can be passed to let or mut fn params

value semantics, **can** be rebound, always* copies
[*]
: if not bound to other var 

## ref

- reference semantics
- passing 
