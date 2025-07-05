# Byte Pair Encoding Tokenizer

Minimal implementation of BPE tokenizer

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
print(f"Decoded: {ids} -> {decoded}")
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
