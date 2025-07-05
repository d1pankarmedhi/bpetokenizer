from collections import defaultdict

class BPETokenizer:
  def __init__(self):
    self.vocab_size = 0
    self.vocab = defaultdict(int)
    self.merges = defaultdict(int) # {(int, int): int}
    self.special_tokens = defaultdict(int) # {str: int}
    self.vocab = self._build_vocab()


  def stats(self, ids):
    """
    Get a map of pairs to frequency for given token ids.
    """
    pairs = {}
    for i in range(len(ids) - 1):
      pair = (ids[i], ids[i+1])
      pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

  def merge(self, ids, pair, index):
    """
    Replace consecutive occurance of pair with given token index.
    """
    updated_ids = []
    i = 0
    while i < len(ids):
      if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
        updated_ids.append(index)
        i += 2
      else:
        updated_ids.append(ids[i])
        i += 1
    return updated_ids


  def _build_vocab(self):
    """
    Build a vocabulary from given corpus.
    """
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for pair, idx in self.merges.items():
      vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    for token, idx in self.special_tokens.items():
      vocab[idx] = token.encode("utf-8")
    return vocab


  def train(self, vocab_size, corpus):
    """
    Train a BPE tokenizer on given corpus.
    """
    assert vocab_size >= 256
    self.vocab_size = vocab_size
    num_merges = self.vocab_size - 256

    # input text processing
    text_bytes = corpus.encode("utf-8") # raw bytes
    ids = list(text_bytes) # list of integers -> 0..255

    # merge most common pairs to create new tokens
    merges = {} # {(int, int): int}
    vocab = self._build_vocab()
    for i in range(num_merges):
      stats = self.stats(ids) # count consecutive pairs
      pair = max(stats, key=stats.get) # find pair with highest count
      idx = 256 + i
      ids = self.merge(ids, pair, idx) # replace consecutive pairs with idx
      self.merges[pair] = idx # save merge
      vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

      print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurances")

    self.vocab = vocab
    self.merges = merges

  def encode(self, text):
    """
    Encode a string into a list of integers.
    """
    text_bytes = text.encode("utf-8")
    ids = list(text_bytes)
    while len(ids) >= 2:
      stats = self.stats(ids)
      pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) # inf for no more merges
      if pair not in self.merges:
        break 
      idx = self.merges[pair]
      ids = self.merge(ids, pair, idx)
    return ids


  def decode(self, ids):
    """
    Decode a list of integers into a string.
    """
    text_bytes = b"".join([self.vocab[idx] for idx in ids])
    return text_bytes.decode("utf-8", errors="replace")