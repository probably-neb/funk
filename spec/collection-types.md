## struct

```
let Person = struct {
    first_name: str
    last_name: str
    age: int
    
    fun full_name() {
        return first_name + ' ' + last_name
    }
}
```

## enum

```
let Color = enum {
    Red
    Yellow
    Blue
    Green
}

assert_false(Color.Red == Color.Blue)

fun only_accepts_primary_colors(color: Color.Red | Color.Blue | Color.Green) {
    # do something
}
```

## union

NOTE: could be combined with enum like rust

```
let Option = union(T) {
    Some(T)
    None
}

let Expr = union {
    Literal(str)
    Binop {
        op: Op,
        lhs: Expr
        rhs: Expr
    }
}
```

## ? record

Essentially an anonymous struct. Allows for quickly mocking something out

NOTE: could be replaced with anonymous struct literals

FIXME: this example sucks

```
fun get_session() {
    let Some(user_id) = get_current_user_id() else {
        return None
    }
    return Some(record {
        user_id
    })
}
```
