let TrieNode = struct {
    ch: u8,
    endNode: bool,
    count: uint,
    children: Option(Self)[26]

    fun new(ch: u8): Self {
        Self {
            ch,
            endNode: false,
            count: 0,
            // FIXME: how to handle default constructors?
            children: Option(Self)[26]
        }
    }
    fun next(mut self, ch: u8): Self {
        let idx = ch - 'a'
        mut child = self.children[idx]
        if child is None {
            child = Some(Self.new(ch))
            self.children[idx] = child
        }
        return child
    }
}

let WordCountNode = struct {
    count: uint,
    word: string,
    next: Option(Self)
    
    fun new(count: uint, word: string, next: Option(Self)): Self {
        Self {
            count,
            word,
            next
        }
    }

    fun is_empty(self): bool {
        let hasNext = match self.next {
            Some(w) => w != self,
            None => false
        }
        return !hasNext
    }
    
    fun swap(mut self, mut other: Self) {
        let word = self.word
        let count = self.count
        self.word = other.word
        self.count = other.count
        other.word = word
        other.count = count
    }

    fun sort(mut self) {
        mut cur = self
        while self.next is Some(next) and next != self {
            // min sort
            if next > self {
                break
            }
            self.swap(next)
            cur = next
        }
    }

    fun length(self): uint {
        mut count = 0
        mut cur = self
        while cur.next is Some(next) {
            count += 1
            cur = next
        }
        return count
    }

    fun pop(mut self) {
        if self.next is Some(next) {
            self.swap(next)
            self.next = next.next
        }
    }

    fun add(mut self, count: uint, word: string, n: int) {
        // insert at pos of self, without needing to return
        // the new root node (self is still root)
        let new = Self.new(count, word, self.next)
        self.next = Some(new)
        self.swap(new)

        self.sort()
        if self.length() > n {
            self.pop()
        }
        return
    }
}

let Cmp = use("std.ops").Cmp

let Cmp(WordCountNode) = trait {
    fun cmp(self, other: Self): Cmp.Order {
        if self.count < other.count {
            return Cmp.Order.Less
        } else if self.count > other.count {
            return Cmp.Order.Greater
        } else {
            return Cmp.Order.Equal
        }
    }
}

const Reader = use("std.io").Reader
const Ascii = use("std.ascii")

fun count_word_frequencies(n uint, inputs: Reader[]): WordCountNode {
    // specify type when using None?
    let counts = None(WordCountNode)

    let trie_root = TrieNode.new(0)
    mut cur_trie_node = trie_root

    // total in fw.c
    mut num_unique_words = 0
    mut cur_word = ""

    for input in inputs {
        while input.read_byte() is Ok(Some(mut ch)) {
            if Ascii.is_alphabetic(ch) {
                ch = Ascii.to_lower(ch)
                cur_word += ch
                cur_trie_node = cur_trie_node.next(ch)
                continue
            }
            cur_trie_node.endNode = true
            if cur_trie_node.count == 0 {
                num_unique_words += 1
            }
            cur_trie_node.count += 1
            match counts {
                None => counts = Some(WordCountNode.new(cur_trie_node.count, cur_word, None)),
                Some(counts) => counts.add(cur_trie_node.count, cur_word, n)
            }
            // reset state
            cur_word = ""
            cur_trie_node = trie_root
        }
    }

    let results = WordCountNode.new(num_unique_words, "", counts)
    return results
}

let left_pad = use("std.str").left_pad

fun print_results(counts: WordCountNode) {
    mut total = counts.count
    mut cur = counts.next
    let n = match cur {
        Some(c) => c.length(),
        None => 0
    }
    print(f"The top {n} words (out of {total}) are:")
    while cur is Some(c) {
        print(f"{left_pad(c.count, 9)} {c.word}")
        cur = c.next
    }
}

let flags = use("flags")

let parser = flags.Parser.new()

parser.add_arg("n", flags.IntArg {
    long: "--top-n",
    short: "-n",
})

parser.add_arg("inputs", flags.PositionalArg {
    multiple: true,
    required: false,
})

let io = use("std.io")

fun main() {
    let args = parser.parse_args()
    let n = match args.n {
        Some(n) => n,
        None => 10
    }
    let inputs = match args.get("inputs") {
        Some(inputs) => inputs.map(fun(input) { io.open(input)}),
        None => [io.stdin]
    }
    defer {
        for input in inputs {
            input.close()
        }
    }
    let counts = count_word_frequencies(n, inputs)
    print_results(counts)
}
