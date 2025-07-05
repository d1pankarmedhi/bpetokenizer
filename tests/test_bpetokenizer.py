import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bpetokenizer import BPETokenizer


def test_bpetokenizer():
    text = "banana banannana"
    bpe = BPETokenizer()
    bpe.train(259, text)
    ids = bpe.encode(text)
    assert ids == [98, 97, 110, 97, 110, 97, 32, 98, 97, 110, 97, 110, 110, 97, 110, 97]
    assert bpe.decode(ids) == text

    print(f"Encoded: {text} -> {ids}")
    print(f"Decoded: {ids} -> {text}")


if __name__ == "__main__":
    test_bpetokenizer()
