# Key Points:
- fast to write
    - but without common foot-guns of python such as "do I need to copy?" and lack of type safety
- fast compile & interp 
    - can't have compile times as slow as rust or it looses viability as scripting language
- ability to "compile" to binary

# Thoughts:
- may be better to switch to actual syntax now
    avoid:
        - too much specialization towards non-explicit control flow
        - specialization for s-exp syntax/expressions
        - polish notation 
            - could still do polish notation for now then switch too non polish after adding operator precedence (pratt)
- In my initial doc I said I'd probably use pointers for mutation like go, I'm no longer certain this is the right course of action
  I like the idea of following rust instead with a mut keyword but I wonder if it the required mental overhead is worth it
- could have ability to turn off type checking for interp speed (when writing quick script)
- want ability to "compile" to a binary
    - maybe just bundle interpreter with byte code?
    - "compile" to go (or other language) for actual compilation
    - support interpretation + compilation (probably very hard!)
        - removes ability to be dynamic (no eval)

# Plans:

this weekend
: add function syntax parsing in lisp notation + implement direct interpretation off of ast. Then transition to non s-exp syntax
next week - investigate set based types (elixir) + other type systems (rust, go, zig, typescript) then begin implementing my own
