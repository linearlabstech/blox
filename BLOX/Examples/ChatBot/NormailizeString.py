
import re
import random
import re
import os
import unicodedata
import codecs
class unicodeToAscii:
    def __call__(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

class Normalize:

    def __init__(self):pass

    def __call__(self,s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s