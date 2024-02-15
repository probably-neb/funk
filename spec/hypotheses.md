The traditional benefits of a scripting language as I see it is that it gets out of the programmers way, and allows them to do something quick and dirty.
My overall hypothesis is that the ability to do something quickly does not have to come at the cost of correctness, or features typically associated with
static, compiled languages. I.e. A scripting language does not need to be dynamic, and in fact a well designed static language can achieve most if not
all of the benefits of a dynamic language with less downsides.

# Hypothesis 1

## Runtime errors can be just as concerning in scripts as they are in applications

While runtime errors/crashes may not take down a service for millions of users in a local script,
there are many times a script is doing something important, such as a database migration, or renaming/moving
a set of files. In these cases, when using a language with exception-based errors, the programmer must defensively
guard against places an error could hypothetically occur (it is also worth noting they must know where exceptions could possibly be thrown from which isn't necessarily the case)
This defensive programming and trying to guess where errors can occur contradicts the supposed benefits of a scripting language. Namely being able to go fast

## Sub-Hypothesis 1.1

### Compile time errors are better than runtime errors for scripts as well as applications

I would much rather know that my check to see if some side-effect completed successfully will cause a panic because of a typo in the function name than finding out after side
effects have completed and now I must decide whether to extract my checking code and run it by itself, run the entire operation again, or manually check that everything went ok

## Sub-Hypothesis 1.2

### Immediate feedback in your editor is better than immediate feedback at runtime

I would rather see a red squiggly in my editor, saying a function exists, than find out when the runtime tries to find the non-existant function and fails.

# Hypothesis 2

## Modern compilers can be fast enough so that a "scripting" language can be compiled without losing the feeling of immediate feedback

Modern compilers can be extremely fast (see Zig, Go). Therefore there is no reason a language aimed at scripting tasks cannot include features typically associated with "compiled"
languages such as a sound type system, static type analysis (and the resulting compile time type errors), type inference, optimization passes, and the ability to create binaries

## Sub-Hypothesis 2.1

### Modern compilers can be fast at generating fast runtime code

Again, see Zig and Go. I am claiming here that one does not have to choose between a slow build and fast runtime or a fast build and slow runtime. Fast build and fast *enough* runtime is
acceptable especially for a scripting language

## Sub-Hypothesis 2.2

### Compilers are fast enough that the types and semantics of the language can (and should) be used for potential optimizations

I.e. Type checking at compile time rather than runtime, optimistic in-place mutations, tail-call optimization, etc.

# Hypothesis 3

## Obscuring the semantics of the language through abstraction does not allow a programmer to write a script faster, a well designed language does

This point is basically a critique of Python. I claim that a language that has clear syntax and semantics around mutability may slow the programmer down
and force them to think things through more than one that doesn't, but it saves having to figure out the semantics later without any help from the language.
I.e. Figuring out where a defensive copy should go. I also claim that the overhead required to encode the semantics of the program within the program, is equal
to or less than the overhead of having to encode them some other way (i.e. Through comments, documentation, or word of mouth), trying to figure them out later, or defensively guarding
against possible errors at all times (which often incurs a runtime overhead!). An example of this would be JavaScript. While declaring a variable is `const` is admittedly not that informative 
(it only prevents the identifier from being rebound not the value from being mutated), it is still regarded as best practice to declare identifiers that won't be rebound as 
`const` as anyone reading it later is immediately aware that they don't have to worry about the identifier being something completely different.

# Hypothesis 4

## A language does not have to be functional or procedural

This hypothesis seems to be more or less proven. Most popular langauges nowadays have functions as first class values, some concept of iterators and anonymous functions, 
and many languages as well as frameworks utilize the benefits of immutability guarantees to some extent.
