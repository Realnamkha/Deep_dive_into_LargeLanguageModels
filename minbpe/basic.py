from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        