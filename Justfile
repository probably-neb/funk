watch *ARGS:
    ls --color=never ./src/**.rs | entr -rc cargo test --tests {{ARGS}}
