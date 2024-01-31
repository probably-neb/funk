#include <ctype.h>
#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DEFAULT_N 10
#define ALPHABET_LENGTH 26
#define TRUE 1
#define FALSE 0
#define WORDCAP 25 /*inital size for words*/
#define NEWKIDS 5

/*
 * for some reason gcc is throwing an error claiming i'm doing
 * implicit declaration of getopt. This is bandaid fix
 */
int getopt(int argc, char *const argv[], const char *options);

/*
 * Used in a Trie data structure that is created
 * as the inputs are read.
 * if it is the end of a word endNode is set to 1
 * and count is updated
 */
struct TrieNode {
    char ch;
    int endNode; /*used as bool*/
    int count;
    int numKids;
    struct TrieNode **kiddies; /*pointers to child nodes */
};
typedef struct TrieNode TrieNode;

/*
 * used in a circularly linked list to keep track of
 * current top words while reading input
 */
struct WordCountNode {
    int count;
    char *word;
    struct WordCountNode *prev;
    struct WordCountNode *next; /* linked node */
};
typedef struct WordCountNode WordCountNode;

/* default constructor for WordCountNode */
WordCountNode *initializeWordCountNode(int count, char *word,
                                       WordCountNode *next,
                                       WordCountNode *prev) {
    size_t wordLen;
    WordCountNode *w = (WordCountNode *)malloc(sizeof(WordCountNode));
    /* copy word to node */
    if (word != NULL) {
        wordLen = (strlen(word) + 1) * sizeof(char);
        w->word = (char *)malloc(wordLen);
        w->word = (char *)memcpy(w->word, word, wordLen);
        /* w -> word = word; */
    } else {
        w->word = NULL;
    }
    w->prev = prev;
    w->next = next;
    w->count = count;
    return w;
}

int isListEmpty(WordCountNode *w) {
    /* booleans for if the corresponding node is valid */
    int prev, next;
    if (w) {
        prev = w->prev != NULL && w->prev != w;
        next = w->next != NULL && w->next != w;
        /* return true if it is empty */
        return !(prev || next);
    }
    /* else */
    perror("Cannot check if an uninitialized list is empty.");
    exit(1);
}

void popPrevNode(WordCountNode **w) {
    WordCountNode *prev = NULL, *newPrev = NULL;

    prev = (*w)->prev;
    if (prev == (*w)) {
        perror("Cannot pop node from empty list.");
        exit(1);
    }
    newPrev = prev->prev;
    newPrev->next = (*w);
    (*w)->prev = newPrev;
    /* this is technically uneeded but will make sure it always breaks
     * if I did something wrong */
    prev->next = NULL;
    prev->prev = NULL;

    free(prev->word);
    free(prev);
    return;
}

/*
 * compares word nodes
 * throws an error if either of its parameters are null
 * or if one of the parameters has a count of -1
 * (i.e. it is the root node)
 * returns > 0 if a > b and < 0 if a < b
 * returns 0 if the nodes words are the same
 * compares words first by count then lexicographically
 * returning the greater of the two
 */
int compWordNode(WordCountNode *a, WordCountNode *b) {
    int comp;
    if (a == NULL || b == NULL)
        error(1, ENOTSUP, "Cannot compare null nodes");
    else if (a->count == -1 || b->count == -1)
        error(0, ENOTSUP, "Cannot compare to root Node");
    comp = strcmp(a->word, b->word);
    if (comp == 0)
        return 0;
    else if (a->count > b->count || (a->count == b->count && comp > 0)) {
        return 1;
    } else
        return -1;
}

/* swaps data of two nodes */
void swap(WordCountNode *a, WordCountNode *b) {
    int tCnt = a->count;
    char *tWrd = a->word;
    /* printf("%s <-> %s", a->word,b->word); */
    a->count = b->count;
    a->word = b->word;
    b->count = tCnt;
    b->word = tWrd;
}

/* sorts using compWordNode comparator
 * swapping as necessary */
void sortWordList(WordCountNode **root) {
    WordCountNode *cur;
    int comp;
    for (cur = (*root)->prev; cur != (*root) && cur->prev != (*root);
         cur = cur->prev) {
        if ((comp = compWordNode(cur, cur->prev)) > 0)
            swap(cur, cur->prev);
        else if (comp == 0)
            perror("Wow this is bad. How could these nodes be equal");
    }
}

/*
 * accepts a count and a word to insert into the wordlist
 * if it finds a node with the same word it updates it's count
 * and continues
 * if the count is >= to the count of the current lowest node
 * it creates a new node and adds it to the end
 * then it runs sortWordList
 * finally, pops the lowest node if it added a new node
 *
 * length of passed list is promised to have a length <= n
 * after this function has completed
 */
void tryAddToTopWordList(WordCountNode **root, int count, char *word, int n) {
    WordCountNode *new = NULL, *cur = NULL;
    int len = 0, added = FALSE;

    for (cur = (*root)->next; cur != (*root); cur = cur->next) {
        if (strcmp(cur->word, word) == 0) {
            cur->count = count;
            added = TRUE;
            len++;
            break;
        }
        len++;
    }

    if (!added && (len < n || count >= (*root)->prev->count)) {
        new = initializeWordCountNode(count, word, (*root), (*root)->prev);
        (*root)->prev->next = new;
        (*root)->prev = new;
        len++;
    }
    sortWordList(root);

    if (len > n) {
        popPrevNode(root);
    }
}

/* resize curWord and curWordLen */
void increaseWordCap(int *curWordPtrLenCap, char **curWordPtr) {
    *curWordPtrLenCap = *curWordPtrLenCap + WORDCAP;
    /*return in case realloc moves it*/
    *curWordPtr =
        (char *)realloc(*curWordPtr, (*curWordPtrLenCap) * sizeof(char));
}

/*
 * resets the current word
 * frees the words memory
 * sets the pointer to null
 * and sets wordcap to WORDCAP
 */
void resetCurWord(int *curWordPtrLenCap, char **curWordPtr) {
    if (curWordPtr != NULL && *curWordPtr != NULL) {
        free(*curWordPtr);
        *curWordPtr = NULL;
    }
    if (curWordPtrLenCap != NULL)
        *curWordPtrLenCap = WORDCAP;
    *curWordPtr = (char *)calloc(*curWordPtrLenCap, sizeof(char));
}

/* appends a character to the current word resizing if necessary */
void addToWord(int *curWordPtrLenCap, int *curWordPtrLen, char **curWordPtr,
               int ch) {
    if (curWordPtr == NULL || *curWordPtr == NULL)
        resetCurWord(curWordPtrLenCap, curWordPtr);

    (*curWordPtr)[*curWordPtrLen] = ch;
    (*curWordPtrLen)++;
    (*curWordPtr)[*curWordPtrLen] = '\0';

    if (*curWordPtrLen == (*curWordPtrLenCap - 1))
        increaseWordCap(curWordPtrLenCap, curWordPtr);
}

/* trie node constructor */
/* notably initializes kids as NULL */
TrieNode *constructTrieNode(int ch) {
    TrieNode *t = (TrieNode *)malloc(sizeof(TrieNode));
    t->endNode = FALSE;
    t->count = 0;
    t->numKids = 0;
    t->ch = ch;
    t->kiddies = NULL; /*  (TrieNode **) malloc(10*sizeof(TrieNode *));*/
    /* assign all nulls to kiddies */
    /* t -> kiddies = NULL; */
    return t;
}

/* gets next trienode by char and initializes it if
 * it is null, and adds it to the passed nodes kids
 * returns the found or created node*/
TrieNode *getNextTrieNode(TrieNode *curNode, int ch) {
    TrieNode **newKiddies = NULL, *kid = NULL;
    int numKids = 0, i = 0;
    if (curNode != NULL) {
        newKiddies = curNode->kiddies;
        numKids = curNode->numKids;
    }

    if (newKiddies != NULL && numKids != 0) {
        /* will continue if:
         * i < numKids aka within bounds
         * or: the node pointed to at kiddies[i] !=NULL
         * nested i++ ensures i is incremented when kid == NULL
         * so it holds the correct index*/
        for (i = 0; i < numKids && (kid = newKiddies[i++]) != NULL;) {
            if (kid->ch == ch) {
                return kid;
            }
        }
    }
    /* i == numKids will be true when:
     * loop ended when i !< numKids
     * kid == NULL when:
     * the array or the TN pointed too by kiddies[i]
     * has not been initialized yet */
    if (i == numKids || kid == NULL) {
        numKids += 1;
        newKiddies =
            (TrieNode **)realloc(newKiddies, numKids * sizeof(TrieNode *));
        if (newKiddies == NULL)
            perror("Failed to realloc kids array");
        newKiddies[i] = NULL;
        curNode->numKids = numKids;
    }

    kid = constructTrieNode(ch);
    newKiddies[i] = kid;
    curNode->kiddies = newKiddies;
    return curNode->kiddies[i];
}

WordCountNode *countWordFrequencies(int n, char **inputs, int numImputs) {
    TrieNode *trieRoot = constructTrieNode(0);
    WordCountNode *wordCountRoot; /* to be constructed shortly */
    /* initialize current trie node to point at root */
    TrieNode *curTrieNode = trieRoot;
    int inputIndex, ch = 0, curWordLen = 0, curWordLenCap = WORDCAP, total = 0;
    FILE *file;
    char *curWord = NULL, *fileName = NULL;
    resetCurWord(&curWordLenCap, &curWord);

    /* construct word count list root */
    wordCountRoot = initializeWordCountNode(-1, "root", NULL, NULL);
    wordCountRoot->prev = wordCountRoot;
    wordCountRoot->next = wordCountRoot;

    for (inputIndex = 0; inputIndex < numImputs; inputIndex++) {
        if (inputs == NULL)
            file = stdin;
        else {
            fileName = inputs[inputIndex];
            file = fopen(fileName, "r");
            if (file == NULL) {
                fprintf(stderr, "%s: Failed to open file \"%s\"... %s\n", "fw",
                        fileName, strerror(errno));
                continue;
            }
        }
        while ((ch = fgetc(file)) != EOF) {
            /* if in alphabet */
            if (isgraph(ch)) {
                /* normalize characters */
                if (isalpha(ch)) {
                    ch = tolower(ch);
                }
                /* go to next TrieNode */
                curTrieNode = getNextTrieNode(curTrieNode, ch);
                /* add character to current word */
                addToWord(&curWordLenCap, &curWordLen, &curWord, ch);
            }
            /* reached end of word */
            else {
                /* does not run if the length of the current word is zero
                 * and therefore the previous char was not a word character */
                if (curTrieNode != trieRoot && curWord != NULL &&
                    strlen(curWord) > 0) {
                    /* check if zero before incrementing total
                     * to only count unique words */
                    if (curTrieNode->count == 0) {
                        total++;
                    }
                    /* update count */
                    curTrieNode->count++;
                    /* set end of word boolean to true */
                    curTrieNode->endNode = TRUE;
                    /* insert the word into the current list of top words
                     * tryAddToTopWordList will not add it if it is below the
                     * top n words */
                    tryAddToTopWordList(&wordCountRoot, curTrieNode->count,
                                        curWord, n);
                    /* jump back to the root for next word */
                    curTrieNode = trieRoot;
                    /* reset the current word */
                    resetCurWord(&curWordLenCap, &curWord);
                    curWordLen = 0;
                }
            }
        }
        if (!(fclose(file) == 0)) {
            perror("Failed to close file\n");
        }
    }
    free(curWord);
    curWord = NULL;

    /* printWordList expects a non circularly linked list
     * who's first node contains the total amount of words considered
     * therefore:
     * set the root nodes count to total */
    sortWordList(&wordCountRoot);
    wordCountRoot->count = total;
    /* and:
     * remove the roots prevs link to the root so it
     * becomes a non circularly linked list */
    wordCountRoot->prev->next = NULL;
    /* return singly linked list of top words with length n */
    return wordCountRoot;
}

/* assumes root node contains total as its count */
void printWordList(const WordCountNode *wordList, int n) {
    int total = wordList->count;
    WordCountNode *cur = wordList->next;

    printf("The top %d words (out of %d) are:\n", n, total);
    while (cur != NULL && cur->word != NULL && cur->count != 0) {
        printf("%*d %s\n", 9, cur->count, cur->word);
        cur = cur->next;
    }
}

/*
 * Below are helper functions I wrote for debugging.
 */

int asprintf(char **ptr, const char *template, ...);

char *printList(WordCountNode *w) {
    WordCountNode *c;
    char *output = "";
    if (w == NULL) {
        perror("cannot print null list");
        exit(1);
    }

    c = w;
    do {
        if (c != NULL) {
            if (asprintf(&output, "%s [%d : %s] <p=n>", output, c->count,
                         c->word) > 0)
                c = c->next;
            else
                return NULL;
        }
    } while (c != NULL && c != w);
    if (asprintf(&output, "%s [%d : %s]\n", output, c->count, c->word) > 0)
        return output;
    else
        return NULL;
}

int lenCircularList(WordCountNode *head) {
    WordCountNode *current;
    int count = 0;
    if (head == NULL) {
        perror("Cannot calculate the length of a non initialized list");
    }
    current = head->next;
    while (current != NULL && current != head) {
        count++;
        current = current->next;
    }
    return count;
}

int main(int argc, char *argv[]) {
    const char *options = ":n:";
    int i, n, c;
    char *nval, *tail, *fileName;
    const char *usagestr =
        "Usage:\n\tfw [-n num] [ files ...]\nOptions:\n\t-n\tSet the number of "
        "most frequent words to display. Defaults to 10.\n\tfiles\tThe files "
        "to read words from. Defaults to reading from stdin.";
    extern char *optarg;
    extern int optopt, errno, optind;
    char **inputs = NULL;
    int numImputs = 1;
    WordCountNode *topWordsList = NULL;

    /* ARGUMENT HANDLING */
    c = getopt(argc, argv, options);
    switch (c) {
    case 'n':
        nval = optarg;
        errno = 0;
        n = strtol(nval, &tail, 0);
        /*if errno was set there was an overflow*/
        if (errno) {
            error(1, errno,
                  "Overflow. Option `-n` requires a smaller argument "
                  "value.\n\n%s",
                  usagestr);
            error(1, errno, "Failed to Parse argument for Option `-n`\n\n%s",
                  usagestr);
        }
        /*if tail doesn't point to the end of the string something went wrong
         * as stated by GNU C library documentation */
        else if (!(*tail == '\0')) {
            error(1, errno, "Option -n requires an integer argument.\n\n%s",
                  usagestr);
        }
        /*at this point we are confident n is an integer*/
        else if (n < 0) {
            error(1, errno,
                  "Option -n requires a non-negative integer argument.\n\n%s",
                  usagestr);
        }
        /* now we're sure n is a valid arg */
        break;
    case '?':
        // TODO: checking for `--help`
        if (optopt == 'h') {
            printf("%s\n", usagestr);
            exit(0);
        }
        error(1, errno, "Unknown option `-%c'.\n\n%s", optopt, usagestr);
    default:
        if (optopt == 'n')
            error(1, errno, "Option -n requires an argument.\n\n%s", usagestr);
        else
            n = DEFAULT_N;
    }
    /*no files passed as args*/
    if (argc == optind) {
        inputs = NULL;
    }
    /*LOAD input FILES*/
    else {
        numImputs = argc - optind;
        inputs = (char **)malloc(sizeof(char *) * numImputs);
        for (i = optind; i < argc; i++) {
            fileName = argv[i];
            inputs[i - optind] = fileName;
        }
    }

    topWordsList = countWordFrequencies(n, inputs, numImputs);
    printWordList(topWordsList, n);
    topWordsList = NULL;

    /*SHUTDOWN*/
    free(inputs);
    inputs = NULL;
    return 0;
}
