# goals

- make mutability and places where mutation occurs obvious
- allow for compiler optimizations based on value mutability

# considerations

`let` => immutable
`mut` => mutable

# swift

`classes` = refs
- always heap allocated
- "value" is actually ref
i.e.
```swift
class A {
    foo: Int
}
let a = A(foo: 0); // let = immutable
let b = a;   // _ref_ to a copied
// allowed because the immutability of b only applies
// to the value of the identifier (i.e. the ref) which does't change
b.foo = 1;
a.foo == 1; // true!
```

# questions

- are `ref` types actually required?
- how much mental overhead is required to understand `ref` semantics
- how will `mut` and `ref` interact with coroutines
