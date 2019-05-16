# distutils: language=c++
# cython: language_level=3

########
# TODO #
########
#
# (1) build discard probability table for negative sampling loss

import os
import pickle

# C/C++ stdlib types/functions
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map as c_map
from libcpp.algorithm cimport sort as c_sort
from cython.operator cimport dereference as deref, preincrement as inc

# numpy to underpin memoryview for word index
import numpy as np
cimport numpy as np

#some type definitions
ctypedef np.int_t int_t
ctypedef c_map[int, int] prune_map


# define some constants
cdef long MAX_VOCAB_SIZE = 30000000
cdef int MAX_LINE_SIZE = 1024
cdef string EOS = b'</s>'
cdef string BOW = b'<'
cdef string EOW = b'>'


# enumerated type for word vs. label distinction
# allows labels to be embedded in same vector space as words
cdef enum entry_type:
    word = 0
    label = 1


# C-struct for word entries. store values:
# (string) word: word value, encoded as binary C-string
# (int) count: incidence count of word in vocabulary
# (entry_type) word_type: word/label distinction in vocabulary
# (vector[int]) subwords: indices of subword strings for word (if maxn, minn > 0)
cdef struct WordEntry:
    int count
    entry_type word_type
    string word
    vector[int] subwords


# supply some helper functions
cdef unsigned int fhash(string word):
    """reimpliment FastText's internal hash function.

    ARGS:
        :param (string) word:
            input word.

    RETURNS:
        :param (unsigned int):
            32-bit hashed value.
    """
    cdef unsigned int h = 2166136261
    cdef size_t i
    for i in range(word.size()):
        h = h ^ <unsigned int>word[i]
        h = h * 16777619
    return h


cdef bint compare(const WordEntry a, const WordEntry b):
    """Comparator function for sorting word list during thresholding.  will
    always rank words before labels, and higher-count words/labels before
    lower-count values

    _note_ Cython compiler will munge python booleans to C `bint`,
    by x==0 > False, x x>0 > True

    ARGS:
        :param (WordEntry) a:
        :param (WordEntry) b:
            `WordEntry` objects to be compared.

    RETURNS:
        :param (bint):
            Returns truth (nonzero) values for entry `a` ranked before `b`.
    """
    if a.word_type != b.word_type:
        return a.word_type < b.word_type    # words before labels
    return a.count > b.count                # higher-count words first


cdef void sort_words(vector[WordEntry]* words):
    """wrapper to C++ stdlib sort using `compare` fn defined above.

    ARGS:
        :param (vector[WordEntry]*) words:
            pointer to vector of `WordEntry` objects to be sorted in-place.
    """
    c_sort(words[0].begin(), words[0].end(), compare)


# TODO can probably be done faster...
cdef void filter_words(vector[WordEntry]* words,
                       vector[WordEntry]* filtered_words,
                       int word_threshold,
                       int label_threshold):
    """faster filtering for words with count below threshold.  Basically the
    logic of a Python list comprehension.

    ARGS:
        :param (vector[WordEntry]*) words:
            pointer to vector of `WordEntry` objects to be filtered.
        :param (vector[WordEntry]*) filtered_words:
            container for filtered vector of `WordEntry`.
        :param (int) word_threshold:
            min count for word-type entries for filtering
        :param (int) label_threshold:
            min count for label-type entries for filtering
    """
    cdef WordEntry w
    for w in words[0]:
        if (((w.word_type == entry_type.word) and (w.count >= word_threshold))
                or ((w.word_type == entry_type.label) and (w.count >= label_threshold))):
            filtered_words.push_back(w)


cdef class Dictionary:
    """FastText dictionary class, used to vectorize texts into sparse
    representation for subsequent use in learning dense representations.

    Notably, FastText introduces subword information: each word is represented
    as a composition of its substrings (min and max length set by
    hyperparameters) rather than a single entity.  This better incorporates
    common information between similar words, and allows the generation of
    nontrivial vectors for OOV tokens.

    Words and labels are represented by `WordEntry` objects, which store the
    root string, incidence count in the corpus, word type (entry vs. label),
    and a vector of subword indices.  The location of each word entry in its
    vector is stored in an index vector, which is accessed via a hash value
    of the word.  In short, to access a word:

    ```
    hash(word) = int_index >> word2int[int_index] = word_id >> words[word_id] = WordEntry
    ```

    where `word_id` $\in {V}$ is the index used for the vocabulary embedding
    layer.  If subword information is used, each word also stores a vector of
    `word_id` indices corresponding to substring locations in the hashed space.

    _note_: the dictionary uses an efficient 32-bit hash to place
    words in the index space.  The `find` function will avoid collisions for
    word entries, but allows collision for subword strings.

    KWARGS:
        :param (str) label: label prefix indicating sample label
        :param (int) minn: minimum length for substrings.  set 0 to skip subword parse.  Default 0.
        :param (int) maxn: maximum length for substrings.  set 0 to skip subword parse.  Default 0.
        :param (int) word_ngrams: length of word ngrams to process.  Default 2.
        :param (int) min_count: minimum threshold for word count.  Default 0.
        :param (int) min_count_label: minimum threshold for label count.  Default 0.
        :param (int) bucket: max number of hash buckets for subword information.  Default 1000000.
        :param (bool) allow_padding: add padding index to word vectors, such that index 0 is reserved for null words.
            Use to pad sentences to fixed length in input tensors for batch training.
    """
    cdef:
        int nwords_
        int nlabels_
        int ntokens_
        int word_ngrams
        int minn
        int maxn
        int pruneidx_size_
        unsigned int nsentence_
        unsigned int min_count
        unsigned int min_count_label
        unsigned int bucket
        int_t size_     # pre-cast to directly set to self.word2int_
        int_t [:] word2int_
        vector[WordEntry] words_
        string label
        prune_map pruneidx_
        bint allow_padding_
        # double [:] discard_probabilities
        # double neg_sample_scale_factor

    def __init__(self,
                 label=b"__label_",
                 minn=0,
                 maxn=0,
                 word_ngrams=2,
                 min_count=0,
                 min_count_label=0,
                 bucket=1000000,
                 allow_padding=False):
        self.label = label
        self.min_count = min_count
        self.min_count_label = min_count_label
        self.bucket = bucket
        self.minn = minn
        self.maxn = maxn
        self.word_ngrams = word_ngrams
        self.allow_padding_ = allow_padding

        # initial values for other attributes
        self.word2int_ = np.zeros(MAX_VOCAB_SIZE, dtype=np.int) - 1
        self.nwords_ = 0
        self.nlabels_ = 0
        self.ntokens_ = 0
        self.words_ = vector[WordEntry]()
        self.size_ = 0
        self.pruneidx_size_ = -1
        self.pruneidx_ = prune_map()
        self.nsentence_ = 0
        # self.neg_sample_scale_factor = 1e-5   # set this as input parameter

    #############################
    # data handlers and getters #
    #############################

    cdef void _allow_padding(self):
        """Batch learning in Keras requires fixed-width input tensors,
        so we need to zero-pad sentences out to a fixed number of tokens.
        reserves zero index in words for dummy `null_word`.

        Called by Python interface function `allow_padding`.
        """
        if self.allow_padding_:
            print('0 index has been saved for padding')
            return

        # set allow_padding and remake arrays
        self.allow_padding_ = True
        cdef WordEntry null_word
        self.words_.insert(self.words_.begin(), null_word)

        cdef size_t i
        cdef size_t word2intsize = len(self.word2int_)
        cdef int_t each
        for i in range(word2intsize):
            each = self.word2int_[i]
            if each != -1:
                self.word2int_[i] = each + 1

    cdef entry_type get_type(self, string w):
        """given a string, check for word/label distinction using stored
        `self.label` value

        ARGS:
            :param (string) w:
                input word for comparison.

        RETURNS:
            :param (entry_type):
                word/label entry type enum
        """
        return entry_type.label if w.startswith(self.label) else entry_type.word

    cdef entry_type get_type_by_id(self, ssize_t word_id):
        """Get entry type by `word_id` index.  Will throw error for OOV values
        (`word_id` < 0 or `word_id` > vocab size).

        ARGS:
            :param (int) word_id:
                integer index of word (found by hash value).

        RETURNS:
            :param (entry_type):
                word/label entry type enum
        """
        assert word_id >= 0
        assert word_id < self.size_
        return self.words_[word_id].word_type

    cdef string get_word(self, ssize_t word_id):
        """get b-string encoded word by ID.  Wil throw error for OOV values
        (`word_id` < 0 or `word_id` > vocab size).

        ARGS:
            :param (ssize_t) word_id:
                signed integer index of word (found by hash value).

        RETURNS:
            :param (string):
                b-string encoded word.
        """
        assert word_id >= 0
        assert word_id < self.size_
        return self.words_[word_id].word

    cdef string get_label(self, ssize_t label_id):
        """get b-string encoded label by ID.  Will throw error for OOV values
        (`word_id` < 0 or `word_id` > number of stored labels).  Note that
        label are indexed modulo nwords_, e.g. for 1000 words and 2 labels,
        label_id 0 and 1 are the 1000th and 1001st indices in the vocabulary.

        ARGS:
            :param (ssize_t) label_id:
                signed integer index of label.

        RETURNS:
            :param (string):
                b-string encoded label.
        """
        assert label_id >= 0
        assert label_id < self.nlabels_
        return self.words_[self.nwords_ + label_id].word

    cdef void _get_labels(self, vector[int]* indices, vector[WordEntry]* labels):
        """
        """
        cdef ssize_t i
        cdef WordEntry label
        for i in range(self.nwords_, self.nwords_ + self.nlabels_):
            label = self.words_[i]
            indices.push_back(i)
            labels.push_back(label)

    cdef vector[int] get_subwords(self, ssize_t wid):
        """Get subwords associated with word in vocabulary.  Will throw error
        for OOV values (`word_id` < 0 or `word_id` > number of words).

        ARGS:
            :param (int) wid:
                word_id, index for `WordEntry` object in `self.words_`.

        RETURNS:
            :param (vector[int]):
                word indices of subwords associated with that word.
        """
        assert wid >= 0
        assert wid < self.nwords_
        return self.words_[wid].subwords

    ###########################################
    # word processing and vocabulary addition #
    ###########################################

    # general flow for word indexing in vocabulary:
    # (1) hash(word) > integer index, avoiding collisions
    # (2) that index points to a value in a word2int_ array, which stores a
    #       word ID
    # (3) word ID is index of WordEntry in final words_ array

    cdef int find(self, string w):
        """Generate hash-index of word.  This index points to a value in an
        integer array (`word2int_`) storing word IDs.

        index is generated via noncryptographic 32-bit hash of word,
        taken modulo vocabulary size.  This is then checked for collisions:
        if that index is valued -1 (non-occupied) the word is assigned to it,
        while if it is occupied (by a different word) the word index is
        incremented until the next empty slot is found.

        ARGS:
            :param (string) w:
                word to be searched.

        RETURNS:
            :param (int):
                integer hash index for word.
        """
        cdef unsigned int h = fhash(w)
        cdef int word2intsize = len(self.word2int_)
        cdef int word_id = h % word2intsize

        cdef int_t word_int = self.word2int_[word_id]
        while (word_int != -1) and (self.words_[word_int].word != w):
            word_id = (word_id + 1) % word2intsize
            word_int = self.word2int_[word_id]

        return word_id

    cdef int get_id(self, string w):
        """Get word ID from a string (step 2).  This ID indexes a WordEntry
        object in the vector of vocabulary words (`words_`).

        Takes hash index from `find` (step 1) and returns index stored in
        `word2int_` array for that hash index.  Will return -1 for OOV words.

        ARGS:
            :param (string) w:
                word to be searched.

        RETURNS:
            :param (int):
                word ID.
        """
        cdef int h = self.find(w)
        return self.word2int_[h]

    cdef void add(self, string w):
        """Add word to the vocabulary (step 3).

        Accesses word ID from `word2int_` by hash value and increments token
        count.  For OOV words (`word_int == -1`) adds a new word: generates a
        `WordEntry` object with that string, a count of 1, an empty subwords
        array, and the appropriate `entry_type`.  This is then appended to the
        `words_` array, and the appropriate nontrivial index is recorded back
        into `word2int_`.  In the case of an already-seen word, the count in
        the existing `WordEntry` is incremented.

        ARGS:
            :param (string) w:
                word to be added.
        """
        cdef int h = self.find(w)
        cdef int word_int = self.word2int_[h]
        cdef WordEntry w_entry

        self.ntokens_ += 1
        if word_int == -1:
            w_entry.word = w
            w_entry.count = 1
            w_entry.subwords = vector[int]()
            w_entry.word_type = self.get_type(w)
            self.words_.push_back(w_entry)
            self.word2int_[h] = self.size_
            self.size_ += 1
        else:
            self.words_[word_int].count += 1

    cdef void threshold(self, int t, int tl):
        """For large-vocabulary problems, introduce a threshold on word counts
        to only retain terms and labels with sufficiently high incidence.

        Runs `filter_words` to cut off words/labels with counts below
        selected threshold value, then reinstantiates word index list and
        populates for existing words in reverse incidence order.

        ARGS:
            :param (int) t:
                word-count threshold
            :param (int) tl:
                label-count threshold
        """
        cdef WordEntry w_entry
        cdef int h

        cdef size_t word2intsize = len(self.word2int_)
        cdef vector[WordEntry] filtered_words = vector[WordEntry]()

        filter_words(&self.words_, &filtered_words, t, tl)
        self.words_ = filtered_words
        sort_words(&self.words_)

        # reset
        self.size_ = 0
        self.nwords_ = 0
        self.nlabels_ = 0
        self.word2int_ = np.zeros(word2intsize, dtype=np.int) - 1
        for w_entry in self.words_:
            h = self.find(w_entry.word)
            self.word2int_[h] = self.size_
            self.size_ += 1
            if w_entry.word_type == entry_type.word:
                self.nwords_ += 1
            if w_entry.word_type == entry_type.label:
                self.nlabels_ += 1

    cdef void _prune(self, vector[int]* idx):
        """Pruning functionality to strip low-information dimensions
        """
        cdef size_t word2intsize = len(self.word2int_)
        cdef vector[int] pruned_words, pruned_ngrams
        cdef int i, ngram
        cdef size_t j, k

        cdef vector[int].iterator it = idx.begin()
        while it != idx.end():
            i = deref(it)
            if i < self.nwords_:
                pruned_words.push_back(i)
            else:
                pruned_ngrams.push_back(i)
            inc(it)

        c_sort(pruned_words.begin(), pruned_words.end())
        idx[0] = pruned_words

        if pruned_ngrams.size() != 0:
            j = 0
            for ngram in pruned_ngrams:
                self.pruneidx_[ngram - self.nwords_] = j
                j += 1
            idx.insert(idx.end(), pruned_ngrams.begin(), pruned_ngrams.end())
        self.pruneidx_size_ = self.pruneidx_.size()

        self.word2int_ = np.zeros(word2intsize, dtype=np.int) - 1

        j = 0
        cdef WordEntry word
        for k in range(self.words_.size()):
            if ((self.get_type_by_id(k) == entry_type.label) or
                    ((j < pruned_words.size()) and (pruned_words[j] == <int>k))):
                self.words_[j] = self.words_[k]
                self.word2int_[self.find(self.words_[j].word)] = j
                j += 1

        self.nwords_ = pruned_words.size()
        self.size_ = self.nwords_ + self.nlabels_
        self.words_.erase(self.words_.begin() + self.size_, self.words_.end())
        self.init_subword_ngrams()

    ################################
    # subword information handlers #
    ################################

    cdef void push_hash(self, vector[int]* hashes, int word_id):
        """helper function to accomodate quantization.  Given hashed value for
        character or word n-gram string, appends value + `self.nwords_` (to
        prevent collision with word indices) to the supplied vector of hashes.

        ARGS:
            :param (vector[int]*) hashes:
                supplied vector of hash indices for ngrams.
            :param (int) word_id:
                hash ID value of token to append to hashes (will be a character
                or word n-gram).
        """
        if self.pruneidx_size_ == 0 or word_id < 0:
            return
        if self.pruneidx_size_ > 0:
            if self.pruneidx_.count(word_id):
                word_id = self.pruneidx_.at(word_id)
            else:
                return
        hashes.push_back(self.nwords_ + word_id)

    cdef void compute_subwords(self, string w, vector[int]* ngrams):
        """compute subword indices for a given word.  Given a word, will
        generate each substring of length between `self.minn` and `self.maxn`
        (inclusive).  This substring is then hashed modulo `self.bucket` and
        appended to the vector of int indices for the subwords.  Note that the
        indices for subwords are incremented starting at `self.nwords_`, so
        there can be no collision between word and subword indices.  At a
        maximum the dimensionality of the dictionary will then be
        `(MAX_VOCAB_SIZE + self.bucket)`.

        _note_ input words for subword generation will have the special BOW and
        EOW characters applied, such that (for example) the n=4 substring
        "word" and input full word "<word>" exist at different indices.

        ARGS:
            :param (string) w:
                input word.
            :param (vector[int]*) ngrams:
                pointer to a vector of character n-gram int indices, updated in-place.
        """
        cdef size_t i, j
        cdef int h, n
        cdef size_t word_size = w.size()
        cdef string ngram

        for i in range(word_size):
            ngram = b''
            n = 1
            j = i
            while (j < word_size) and (n <= self.maxn):
                ngram.push_back(w[j])
                j += 1
                if (n >= self.minn) and not (n == 1 and (i == 0 or j == word_size)):
                    h = fhash(ngram) % self.bucket
                    self.push_hash(ngrams, h)
                n += 1

    cdef void _compute_subwords_with_strings(self, string w, vector[int]* ngrams, vector[string]* substrings):
        """helper function to compute subword indices plus the accompanying strings.

        Given a word, will generate each substring of length between
        `self.minn` and `self.maxn` (inclusive).  This substring is then hashed
        modulo `self.bucket` and appended to the vector of int indices for the
        subwords.  Note that the indices for subwords are incremented starting
        at `self.nwords_`, so there can be no collision between word and
        subword indices.  At a maximum the dimensionality of the dictionary
        will then be `(MAX_VOCAB_SIZE + self.bucket)`.

        _note_ input words for subword generation will have the special BOW and
        EOW characters applied, such that (for example) the n=4 substring
        "word" and input full word "<word>" exist at different indices.

        Called by Python interface function `compute_subwords_with_strings`.

        ARGS:
            :param (str) w:
                input word.
            :param (vector[int]*) ngrams:
                pointer to vector of character n-gram int indices, updated in-place.
            :param (vector[string]*) substrings:
                pointer to vector of substrings, updated in-place.
        """
        cdef size_t i, j
        cdef int h, n
        cdef size_t word_size = w.size()
        cdef string ngram

        for i in range(word_size):
            ngram = b''
            n = 1
            j = i
            while (j < word_size) and (n <= self.maxn):
                ngram.push_back(w[j])
                j += 1
                if (n >= self.minn) and not (n == 1 and (i == 0 or j == word_size)):
                    h = fhash(ngram) % self.bucket
                    self.push_hash(ngrams, h)
                    substrings.push_back(ngram)
                n += 1

    cdef void add_subwords(self, vector[int]* line, string token, int wid):
        """generate subword indices for a word, and append into a vector of
        indices.  Used by `get_line` to render a window of text into a set of
        indices for its component word and ngram tokens.  Flow:

        if word is not in vocabulary:
            compute subwords as vector of int hashes, and append to the line
            vector.  If no subword information is used (`maxn <= 0`) will
            append an empty vector.
        if word is in vocabulary:
            if no subword information is used, append the word index to the
            line vector.  If using subword information, append the `subwords`
            vector stored for the `WordEntry` object (this will include the
            word itself).

        ARGS:
            :param (vector[int]*) line:
                pointer to vector of int indices representing the window of text.
            :param (string) token:
                word to be added to line.  Only used in the case of OOV words.
            :param (int) wid:
                word ID of word to be added, used to access `WordEntry` storage
                or detect OOV words.
        """
        if wid < 0:  # out of vocab
            if token != EOS:
                self.compute_subwords(BOW + token + EOW, line)
        else:  # inside the vocab
            if self.maxn <= 0:  # in vocab w/o subwords
                line.push_back(wid)
            else:  # in vocab with subwords
                ngrams = self.get_subwords(wid)
                line.insert(line.end(), ngrams.begin(), ngrams.end())

    cdef void init_subword_ngrams(self):
        """initializer to generate subword information for in-vocabulary words.
        Called after vocabulary build stage.  Loops through `self.words_` and
        computes subword vectors for each `WordEntry` object.
        """
        cdef int_t indx
        cdef string w

        for indx in range(self.size_):
            self.words_[indx].subwords.clear()
            w = BOW + self.words_[indx].word + EOW
            self.words_[indx].subwords.push_back(indx)
            if self.words_[indx].word != EOS:
                self.compute_subwords(w, &self.words_[indx].subwords)

    #################################
    # word-level sampling functions #
    #################################

    cdef void add_word_ngrams(self, vector[int]* line, vector[int]* hashes, int n):
        """Compute and append word n-gram indices to line.  Called by `_get_line`
        to generate word n-grams for window of text.

        ARGS:
            :param (vector[int]*) line:
                pointer to vector of int indices corresponding to text window.
            :param (vector[int]*) hashes:
                pointer to vector of indices of base words (not including subwords).
            :param (int) n:
                size of ngram window.  Default value set by `self.word_ngrams` at 2.
        """
        assert n >= 0

        cdef ssize_t hashsize = hashes.size()
        cdef ssize_t i, j
        cdef int window
        cdef unsigned int h
        for i in range(hashsize):
            h = hashes[0][i]
            window = min(i + n, hashsize)
            for j in range(i + 1, window):
                h = h * 116049371 + hashes[0][j] # create a hash result for the word ngram
                self.push_hash(line, h % self.bucket)

    ##############################
    # string interface functions #
    ##############################

    def _word_reader(self, str file):
        """Emits token stream from text file to build vocabulary.
        Defined as python function, as C-func definition in Cython does not
        support generator syntax.  However, this is not intended to be
        directly called from the Python interface.

        ARGS:
            :param (str) file:
                path to text file.

        RETURNS:
            generator of `string` objects
        """
        cdef str line
        cdef list line_list
        cdef str word

        with open(file, 'r') as f:
            for line in f:
                self.nsentence_ += 1
                line_list = line.split()
                for word in line_list:
                    if (word == '\n') or (word == '\r') or (word == '\t') or (word == '\v') or (word == '\f') or (word == '\0') or (word == ','):
                        continue
                    else:
                        yield word.encode('utf8')

    def _string_reader(self, string entry):
        """Emits token stream from string input to translate single line
        to token set.  Defined as a python function, as C-func definition in
        Cython does not support generator syntax.  However, this is not
        intended to be directly called from the Python interface.

        ARGS:
            :param (str) entry:
                input text string to be tokenized.

        RETURNS:
            generator of `str` objects
        """
        cdef string word
        cdef list entry_list = entry.split()

        for word in entry_list:
            if word == b'\n' or word == b'\r' or word == b'\t' or word == b'\v' or word == b'\f' or word == b'\0'or word == b',':
                continue
            else:
                yield word

    def _read(self, reader):
        """Text reader function for vocabulary building.  Reads each word
        emitted by `reader` (which may be from a file or in-memory) and adds
        to the vocabulary, indexing by hash value of the string to store count
        information.  Incorporates thresholding for large vocaulary sets
        (triggered as vocab approaches 75% of max size).  Precomputes subword
        information for words if minn or maxn > 0.

        ARGS:
            :param (generator) reader:
                text reader emitting `string` objects.
        """
        cdef int min_threshold = 1
        cdef string word
        for word in reader:
            self.add(word)
            if (self.ntokens_ % 1000000 == 0):
                print(f'Read {self.ntokens_/1000000:.3f} M words')
            if self.size_ > 0.75 * MAX_VOCAB_SIZE:
                min_threshold += 1
                self.threshold(min_threshold, min_threshold)
                print(f'Approaching MaxVocabSize, thresholding apply with minThreshold = {min_threshold}')

        self.threshold(self.min_count, self.min_count_label)
        # self.init_discard_table()
        if self.minn != 0 or self.maxn != 0:
            self.init_subword_ngrams()

        print(f'Read {self.ntokens_/1000000:.3f} M words')
        print(f'Number of words: {self.nwords_}')
        print(f'Number of labels: {self.nlabels_}')
        if self.size_ == 0:
            print ('Empty Vocabulary. Try a smaller -min_count value.')

    cdef int _get_line(self, string sentence, vector[int]* words, vector[int]* labels):
        """renders input sentence into set of `word_id` indices corresponding
        to vocabulary locations, including subwords, for word and label tokens.
        Use to convert raw input sentence to set of indices for input to
        embedding layer (compatible with keras `Embedding` layer type).

        _note_: the embedding layer should be of dimension `MAX_VOCAB_SIZE + self.bucket`
        to accomodate the maximum number of words + character/word n-gram indices.

        Called by Python interface function `get_line`.

        ARGS:
            :param (str) sentence:
                input sting of sentence, tokenized by `_string_reader`.
            :param (vector[int]*) words:
                pointer to vector of integer indices for words, updated in-place.
            :param (vector[int]* labels):
                pointer to vector of label indices (defined modulo nwords_), updated in-place.

        RETURNS:
            :param (int):
                number of tokens read from sentence.
        """
        cdef string token
        cdef int h, wid
        cdef entry_type word_type
        cdef vector[int] word_hashes

        cdef int ntokens = 0

        reader = self._string_reader(sentence)
        for token in reader:
            h = self.find(token)
            wid = self.get_id(token)
            if wid < 0:
                word_type = entry_type.word
            else:
                word_type = self.get_type_by_id(wid)
            ntokens += 1

            if word_type == entry_type.word:
                self.add_subwords(words, token, wid)
                word_hashes.push_back(h)
            elif (word_type == entry_type.label) and (wid >= 0):
                labels.push_back(wid - self.nwords_)    # last words in the index are the labels

            if token == EOS:
                break

        self.add_word_ngrams(words, &word_hashes, self.word_ngrams)
        return ntokens

    ##############################
    # Python interface functions #
    ##############################

    def get_words(self):
        """returns vector of `WordEntry` objects after vocabulary build.
        The attribute `self.words_` is stored as a C-type `vector[WordEntry]`
        object - as a Python interface function, this is munged to a list of
        dicts (`WordEntry` is defined as a C-struct internally).

        RETURNS:
            :param (list[dict]):
                list of `WordEntry` objects
        """
        return self.words_

    def get_labels(self):
        """Returns vectors of indices and label values for every label in the
        vocabulary.  Will only be populated for supervised models.

        RETURNS:
            :param (list[int]):
                list of indices storing labels in the vocabulary.
            :param (list[dict]):
                list of `WordEntry` objects for the labels in the vocabulary.
        """
        cdef vector[int] indices
        cdef vector[WordEntry] labels
        self._get_labels(&indices, &labels)
        return indices, labels

    def get_nwords(self):
        """Getter method returning the number of words in the vocabulary.

        RETURNS:
            :param (int):
                number of words in the vocabulary.
        """
        return self.nwords_

    def get_nlabels(self):
        """Getter method returning the number of labels in the vocabulary.

        RETURNS:
            :param (int):
                number of labels in the vocabulary.
        """
        return self.nlabels_

    def get_buckets(self):
        """Getter method returning the number of buckets (hash indices for
        character or word n-grams) - the input dimension of the trained model
        will be |nwords + buckets|.

        RETURNS:
            :param (int):
                number of buckets specified in the model.
        """
        return self.bucket

    def get_input_dims(self):
        """returns the input dimension size for the network, defined as one
        index for each word and word/char n-gram.

        RETURNS:
            :param (int):
                input dimension size for the model.
        """
        return self.nwords_ + self.bucket

    def read_from_file(self, str file):
        """Build vocabulary from text file of samples.
        Reads each word emitted by `_word_reader` and adds to vocabulary,
        indexing by hash value of the string to store count information.
        Incorporates thresholding for large vocaulary sets (triggered as vocab)
        approaches 75% of max vocab).  Precomputes subword information for
        words if minn or maxn > 0.

        Thin wrapper around `self._read(generator)`.

        ARGS:
            :param (str) file:
                path to text file.
        """
        wordreader = self._word_reader(file)
        self._read(wordreader)

    def read_text(self, string text):
        """Build vocabulary from in-memory text.
        Reads each word emitted by `_string_reader` and adds to vocabulary,
        indexing by hash value of the string to store count information.
        Incorporates thresholding for large vocaulary sets (triggered as vocab)
        approaches 75% of max vocab).  Precomputes subword information for
        words if minn or maxn > 0.

        Thin wrapper around `self._read(generator)`.

        ARGS:
            :param (str) file:
                path to text file.
        """
        stringreader = self._string_reader(text)
        self._read(stringreader)

    def get_line(self, string sentence):
        """Get set of word indices (including subwords, if set) for input
        sentence.  Thin wrapper to Cython C-func `_get_line`.

        ARGS:
            :param (str) sentence:
                sentence to be tokenized.

        RETURNS:
            :param (int) ntokens:
                number of word tokens.
            :param (list) words:
                list of word indices (including subwords).
            :param (list) labels:
                list of label indices in sentence.
        """
        cdef vector[int] words
        cdef vector[int] labels
        cdef int ntokens

        ntokens = self._get_line(sentence, &words, &labels)
        return ntokens, words, labels

    def compute_subwords_with_strings(self, string w):
        """Compute subwords of word, returning both indices and the
        corresponding substrings.  Thin wrapper to Cython C-func
        `_compute_subwords_with_strings`.

        ARGS:
            :param (str) w: word to be computed.

        RETURNS:
            :param (list) ngrams:
                list of integer indices of subwords in vocabulary.
            :param (list) substrings:
                list of b-strings of subwords.
        """
        cdef vector[int] ngrams = vector[int]()
        cdef vector[string] substrings = vector[string]()
        self._compute_subwords_with_strings(w, &ngrams, &substrings)
        return ngrams, substrings

    def allow_padding(self):
        """if we allow padding, we will leave index 0 to be blank for use in
        padding.  Batch learning in Keras requires fixed-width input tensors,
        so we need to zero-pad sentences out to a fixed number of tokens.

        Thin wrapper to Cython C-function `_allow_padding`.
        """
        self._allow_padding()

    def prune(self, indices):
        """Prune the dictionary down to the selected indices, discarding the
        rest.  This allows us to dramatically reduce the size of the
        dictionary, at the expense of not allowing additional vocabulary to be
        added.

        ARGS:
            :param (iterable) indices:
                iterable of integers, capable of packing into a C++ vector.

        Thin wrapper to the Cython C-function `_prune`.
        """
        cdef vector[int] v = indices
        self._prune(&v)

    #################
    # serialization #
    #################

    def save(self, str filepath):
        """serialize attributes into picklable dictionary.  `vector[struct]`
        instances like `self.words_` are rendered as a list of dicts in Python
        (and can be reconstructed from the same).  Memoryviews like
        `self.word2int_` must be rendered as explicit numpy arrays for
        serialization/deserialization.

        ARGS:
            :param (str) filepath:
                target path for writing save file.
        """
        with open(filepath, 'wb') as wf:
            pickle.dump(
                {
                    "label": self.label,
                    "min_count": self.min_count,
                    "min_count_label": self.min_count_label,
                    "bucket": self.bucket,
                    "minn": self.minn,
                    "maxn": self.maxn,
                    "word_ngrams": self.word_ngrams,
                    "allow_padding_": self.allow_padding_,
                    "word2int_": np.asarray(self.word2int_),
                    "nwords_": self.nwords_,
                    "nlabels_": self.nlabels_,
                    "ntokens_": self.ntokens_,
                    "words_": self.words_,
                    "size_": self.size_,
                    "pruneidx_size_": self.pruneidx_size_,
                    "pruneidx_": self.pruneidx_,
                    "nsentence_": self.nsentence_
                },
                wf
            )

    def load(self, str filepath):
        """Reset attributes of `Dictionary` object from serialized file.
        C++ vectors (like `self.words_`, `vector[struct]`) are rendered as a
        list of dicts in Python, while the memoryview `self.word2int_` must be
        reconstructed from an explicit numpy array object in the serialization.

        ARGS:
            :param (str) filepath:
                path of serialized file to read.
        """
        with open(filepath, 'rb') as rf:
            data = pickle.load(rf)

        self.label = data['label']
        self.min_count = data['min_count']
        self.min_count_label = data['min_count_label']
        self.bucket = data['bucket']
        self.minn = data['minn']
        self.maxn = data['maxn']
        self.word_ngrams = data['word_ngrams']
        self.allow_padding_ = data['allow_padding_']
        self.nwords_ = data['nwords_']
        self.nlabels_ = data['nlabels_']
        self.ntokens_ = data['ntokens_']
        self.words_ = data['words_']
        self.size_ = data['size_']
        self.pruneidx_size_ = data['pruneidx_size_']
        self.pruneidx_ = data['pruneidx_']
        self.nsentence_ = data['nsentence_']

        # little bit of special to handle memoryviews
        tmp = data['word2int_']
        self.word2int_ = tmp
