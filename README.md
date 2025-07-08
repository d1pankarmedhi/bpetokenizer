# Byte Pair Encoding Tokenizer

Minimal implementation of BPE tokenizer

BPE tokenizer merges the most frequent adjacent pairs of characters or existing tokens into new token units. This helps in handling rare words, spelling variants and compond constructions. Language models, such as GPT-2, uses a BPE tokenizer to model all unicode strings, which allows evaluation on any language model benchmak regardless of processing or tokenization.

In the initial stage, the tokenizer assigns the vocabulary with 256 bytes tokens, considering every single possible byte value as its initial token. This means, the tokenizer can represent any sequence of bytes, such as, text, emojis or any other special characters or text from other language or even binary data. This ensures that there is no information loss due to unknown character encountered by the tokenizer.

```bash
# Initial vocabulary with 256 (0..255) bytes
vocab = {0: b'\x00',
        1: b'\x01',
        ...,
        98: b'b',
        99: b'c',
        100: b'd',
        101: b'e',
        ...,
        255: b'\xff'}
```

For an input text, the BPE tokenizer converts the input into integer IDs representing its UTF-8 bytes. After calculating the frequency of every adjacent pair of tokens and the pair with the maximum frequency is picked, then replaced with a new token ID.

```bash
# input text
text = "apple and banana"

# encoded IDs
"apple and banana" -> [97, 112, 112, 108, 101, 32, 97, 110, 100, 32, 98, 97, 110, 97, 110, 97]

# frequency map for each pair
{(97, 112): 1,
 (112, 112): 1,
 (112, 108): 1,
 (108, 101): 1,
 (101, 32): 1,
 (32, 97): 1,
 (97, 110): 3,
 (110, 100): 1,
 (100, 32): 1,
 (32, 98): 1,
 (98, 97): 1,
 (110, 97): 2}
```

If we consider the vocab size of 259, the number of merges performed is (vocab_size - 256) 3. For each iteration, the max frequency pair is replaced with a new token.

```bash
# for 3 merges
merge 1: (97, 110) - 256 -> [97, 112, 112, 108, 101, 32, 256, 100, 32, 98, 256, 256, 97]
merge 2: (97, 112) - 257 -> [257, 112, 108, 101, 32, 256, 100, 32, 98, 256, 256, 97]
merge 3: (257, 112) - 258 -> [258, 108, 101, 32, 256, 100, 32, 98, 256, 256, 97]
```

The final updated vocabulary will have the merged tokens along with their updated bytes value.

```bash
# focused on updated vocabulary only
258: b'app'
108: b'l'
101: b'e'
32: b' '
256: b'an'
100: b'd'
32: b' '
98: b'b'
256: b'an'
256: b'an'
97: b'a'
```

When encoding a new input text, the tokens from the learned vocabulary can be used to obtain the IDs list for that input.

## Installation

Clone this repository and install the requirements (if any):

```sh
git clone https://github.com/d1pankarmedhi/bpetokenizer.git
cd bpetokenizer
pip install -r requirements.txt  # if requirements.txt exists
```

## Usage

```python
from bpetokenizer import BPETokenizer

text = "banana banannana"
bpe = BPETokenizer()
bpe.train(259, text)
ids = bpe.encode(text)
decoded = bpe.decode(ids)

print(f"Encoded: {text} -> {ids}")
# Encoded: banana banannana -> [98, 97, 110, 97, 110, 97, 32, 98, 97, 110, 97, 110, 110, 97, 110, 97]

print(f"Decoded: {ids} -> {decoded}")
# Decoded: [98, 97, 110, 97, 110, 97, 32, 98, 97, 110, 97, 110, 110, 97, 110, 97] -> banana banannana
```

## Project Structure

```
bpetokenizer/
│
├── bpetokenizer.py      # Main tokenizer implementation
├── tests/
│   └── test_bpetokenizer.py  # Unit tests
└── README.md
```

## License

MIT License
