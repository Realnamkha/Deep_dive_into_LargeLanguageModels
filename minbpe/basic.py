import os
import time
from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # input text processing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integer in range 0 - 255
        
        # iteratively merge the most two common pair to create new tokens
        merges = {} # (int,int) -> int
        vocab = {idx:bytes([idx]) for idx in range(256)}
        
        for i in range(num_merges):
            # count the number of times each consecutive pair appears
            stats = get_stats(ids)
            # Get the top pair from stats 
            top_pair = max(stats,key=stats.get())
            # create a new token: assign it in next available id
            idx = 256 + i
            # replace all occurence of top pair in ids with idx
            ids = merge(ids,top_pair,idx)
            # save the merge
            merges[top_pair] = idx
            #update the vocab
            vocab[idx] = vocab[top_pair[0]] +vocab[top_pair[1]]
        
        
        if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {stats[top_pair]} occurrences")
        
        
        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
        
    
    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8",errors="replace")
        return text
    
    
    def encode(self, text):
        # given a string text, return the token ids
        text_byes = text.encode("utf-8") #raw bytes
        ids = list(text_byes) # list of integers in range 0 .... 255
        while (len(ids)>=2):
            # find the pair with lowest merge index
            stats = get_stats(ids)
            pair = min(stats,key=lambda p:self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            #otherwise merge the best pair(lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids,pair,idx)
        return ids


text = open("test.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer], ["basic"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")
        