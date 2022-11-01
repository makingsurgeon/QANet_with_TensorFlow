# coding=utf-8
# changed from Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
from tqdm import tqdm
import numpy as np

# This design was inspired from erfmca's implemntation that was mentioned in the report and also in train.py
def convert_to_unicode(text):
    #Convert a text to unicode
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def printable_text(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def load_vocab(vocab_file, size=int(2.2e6), vec_size=300):
    #Loads a certain amount of vocabulararies into a dictionary.
    vocab = collections.OrderedDict()
    with open(vocab_file, "r") as reader:
        for line in tqdm(reader, total=size):
            array = line.split()
            word = "".join(array[0:-vec_size])
            word = normalize_text(word)
            vocab[word] = -1
    vocab['--NULL--'] = 0
    vocab['--PAD--'] = 1
    return vocab


def convert_tokens_to_ids(vocab, tokens):
    #Convert some tokens into ids
    ids = []
    for token in tokens:
        unk = True
        for each in (token, token.lower(), token.capitalize(), token.upper()):
            if each in vocab:
                unk = False
                ids.append(vocab[each])
                break
        if unk:
            ids.append(vocab['--PAD--'])
    ids = np.array(ids)
    return ids


def whitespace_tokenize(text):
   # Get rid of white spaces and split text to words
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    #This is a more complete tokenizer.

    def __init__(self, vocab_file, lower_case=False):
        self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = BasicTokenizer(lower_case=lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def get_word_embedding(self, word_count, vocab_file, size=int(2.2e6), vec_size=300):
        self.vocab = collections.OrderedDict()
        self.vocab['--NULL--'] = 0
        self.vocab['--PAD--'] = 1
        word_embedding = np.zeros((len(word_count) + 2, vec_size))
        index = 2
        with open(vocab_file, "r") as reader:
            for line in tqdm(reader, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                word = normalize_text(word)
                vector = np.array(list(map(float, array[-vec_size:])))
                if word in word_count:
                    self.vocab[word] = index
                    word_embedding[index, ::] = vector
                    index += 1
                    word_count.pop(word)
        assert index == len(self.vocab)

        return word_embedding

    def get_char_embedding(self, char_count, vec_size=64):
        char_embedding = np.zeros((len(char_count) + 2, vec_size))
        self.char_vocab = collections.OrderedDict()
        self.char_vocab['--NULL--'] = 0
        self.char_vocab['--PAD--'] = 1

        index = 2
        for char in char_count:
            self.char_vocab[char] = index
            char_embedding[index, ::] = np.random.normal(scale=0.1, size=(vec_size))
            index += 1
        assert index == len(self.char_vocab)

        return char_embedding

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens)


class BasicTokenizer(object):
    # This basically tokenizes the text

    def __init__(self, lower_case=True):
        self.lower_case = lower_case

    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        original_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in original_tokens:
            if self.lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        #Gei rid of accents on letters.
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        # Split words based on punctuation
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization, which is a subword-based tokenzation algorithm."""

    def __init__(self, vocab, unk_token="'--PAD--'", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(text)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                current_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    # if start > 0:
                    #     substr = "##" + substr
                    if substr in self.vocab:
                        current_substr = substr
                        break
                    end -= 1
                if current_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(current_substr)
                start = end

            if is_bad:
                output_tokens.append(text)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    # Check if certain characters are whitespace
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
   #Check if the character is a control character
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    #Check whetehra character is a punctuation
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
