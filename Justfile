watch *ARGS:
    ls --color=never ./src/**.rs | entr -rc cargo test --tests {{ARGS}}

watch-trace *ARGS:
    ls --color=never ./src/**.rs | RUST_BACKTRACE=1 entr -rc cargo test --tests {{ARGS}}

cg-annotate *ARGS:
    #!/usr/bin/bash

    before_file=$(mktemp)
    ls --color=never -1 cachegrind.out* > ${before_file}

    cargo build
    valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes target/debug/funk {{ARGS}};

    after_file=$(mktemp)

    ls --color=never -1 cachegrind.out* > ${after_file}
    
    new_file=$(sort ${before_file} ${after_file} | uniq -u)
    rm ${before_file} ${after_file}

    cg_annotate ${new_file}

cg *ARGS:
    cargo build
    valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes target/debug/funk {{ARGS}};

dhat *ARGS:
    cargo build
    valgrind --tool=dhat target/debug/funk {{ARGS}};
