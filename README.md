# fasttext-keras

Python implementation for the [FastText](https://fasttext.cc/) model
implementation developed by Facebook.

Currently we use the compiled binary
distributed by Facebook Research -- while this is very efficiently coded, it
introduces several difficulties: namely, the implementation introduces some
bugs to the batch prediction stage, is highly reliant on repeated string
processing and file I/O during training and prediction, cannot be extended
into larger models (for example, serialized word embedding are frequently used
as an input layer in more complex sequential models), and is CPU-bound for
training and prediction.

to that end, we develop a Python implementation of the model, directly building
the model in a Tensorflow/Keras network.  This gives us greater control over
the implementation, allowing easy extension into more complex deep models,
and allows GPU training and serving.  The model is broken into two major
components: the dictionary and the network.  Broadly, text (broken into $k$
tokens) can be represented as a $k$-hot vector of size $V$, where $V$ is our
chosen vocabulary size.  This can be expressed in a denser format by a vector
of size $k$, each element $w_i$ containing an index ($w_i \in [0, V-1]$)
corresponding to that word.  This is then used to learn the dense embedding
layer, which represents each word as a dense $d$-dimensional vector: in terms
of implementation, the embedding layer represents a $V \times d$ matrix of
weights.

The representation of text into the $V$ indices is accomplished by the
dictionary stage, with the embedding layer subsequently learned by the network
(this allows the same dictionaries/embeddings to be used for different
model architectures).  Beyond simply tokenizing and indexing the words, the
dictionary implemented here allows for both word and character n-grams: that
is, we can generate unique elements in the vocabulary for pairs, trios, etc.
of contiguous words to capture local semantic information, as well as
representing words as a composition of unique tokens for its constituent
subwords -- this allows generation of nontrivial vectors for out-of-vocabulary
words, as well as encouraging closer vector similarity for
compositionally-related words.  See [here](https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3) for more details on the implementation.

### building

To build locally from git cloning, run

```
python setup.py build_ext --inplace
```

which will build the Cython extensions into importable modules inplace - the
package can then be imported (provided the project directory is in the search
path) without installing (useful for development purposes).  To install,

```
python setup.py bdist_wheel
```

will build a Python 3.6-compatible wheel file from the package under the
`PyFastText/dist/` directory, which can then be installed via pip or uploaded
through twine.  Alternately,

```
python setup.py install
```

will directly install the package into the current environment.

__TODO__
* documentation compatible with mkdocs
* unit testing

## Dictionary

Vocabulary building is often a lengthy process, requiring building up a large
dictionary of tokens.  To speed this up, we implement the dictionary in Cython
to closely mirror the original C++ implementation used by `FastText`.  Given
the training text, the dictionary builds up an indexed vocabulary of terms,
storing the word, its constituent subwords, incidence count, and token type
(distinguishing words vs. labels for supervised training).  These are indexed
via a hashing trick: that is, we store two arrays, an indexer array and the
words array, and access words via the flow

```
hash(word) = ix >> indexer[ix] = word_id >> words[word_id] = word
```

enabling efficient storage and search.  In addition to storing the word
vocabulary (out to size `MAX_VOCAB_SIZE`), this also allows indices (up to
`buckets` additional values -- unused indices are stored as `-1` to skip
mapping to an in-vocabulary word) for word and character n-grams.  To build
representation vector of a given window of text, the dictionary applies the
indices of each token in the vocabulary, as well as the set of subword
indices for each word (if applicable).  OOV words are represented solely by
their component subwords.

### Usage

```
from pyfasttext.dictionary import Dictionary
d = Dictionary()
```

with optional input arguments for bucket size, label prefix, and word/char
n-gram behavior.  To read from a text file,

```
d.read_from_file(<path_to_file>)
```

which will build the vocabulary.  The set of words and labels is accessible via
`d.get_words()` and `d.get_labels()` respectively (note that the return from
`get_words` is often extremely long!).  To process a string text window to a
set of indices, run `d.get_line(<string to process>)`.  Finally, dictionaries
can be saved and loaded via `d.save(<save file>)` and `d.load(<save file>)`.

__TODO__

* build discard probability table to support negative sampling loss
* pruning & quantization support
* enable vocabulary build task from in-memory or streamed text, rather than
read-from-file
* improve file I/O parse?  currently barely better than bare Python speed, is
bottleneck.

## Network Training

__TODO__
* hierarchical softmax, negative sampling loss functions
* ...the rest of the network

### Supervised Training

 __TODO__
 * helper functions for building supervised training setup

### CBOW

__TODO__

### Skip-gram

__TODO__
