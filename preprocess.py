from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from utils import tokenization
import numpy as np
import pickle
import collections

# This design was inspired from erfmca's implemntation that was mentioned in the report and also in train.py
def improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         original_answer_text):
    # Sometimes the ground truth is a substring of a word in the context, and we want to find the best answer.
    # The following function deals with this problem
    tok_answer_text = " ".join(tokenizer.tokenize(original_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def check_is_max_context(doc_spans, cur_span_index, position):
    # We want to find out which span contains the most context, which is defined as the minimum of the amount of
    # words to the left and right of a particular word.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class SquadExample(object):
   #Construct a single sample.

    def __init__(self,
                 uuid,
                 question_text,
                 doc_tokens,
                 original_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.uuid = uuid
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.original_answer_text = original_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "uuid: %s" % (tokenization.printable_text(self.uuid))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
   #Features in the sample.

    def __init__(self,
                 qid,
                 uuid,
                 doc_span_index,
                 token_to_original_map,
                 token_is_max_context,
                 doc_tokens,
                 ques_tokens,
                 start_position=None,
                 end_position=None):
        self.qid = qid
        self.uuid = uuid
        self.doc_span_index = doc_span_index
        self.token_to_original_map = token_to_original_map
        self.token_is_max_context = token_is_max_context
        self.doc_tokens = doc_tokens
        self.ques_tokens = ques_tokens
        self.start_position = start_position
        self.end_position = end_position


# first preprocess to get tokens
def read_squad_examples(input_file, is_training):
    total = 0
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in tqdm(input_data):
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                uuid = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                original_answer_text = None
                if is_training:
                    if len(qa["answers"]) > 1:
                        raise ValueError("For training, each question should have exactly 0 or 1 answer.")
                    answer = qa["answers"][0]
                    original_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(original_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(tokenization.whitespace_tokenize(original_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        print("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                        continue

                total += 1

                example = SquadExample(
                    uuid=uuid,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    original_answer_text=original_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length=384, doc_stride=128, max_query_length=64, is_training=True):
    #Loads data into batches.

    qid = 0
    features = []
    for example_index, example in enumerate(tqdm(examples)):
        uuid = example.uuid
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_original_index = []
        original_to_tok_index = []
        all_doc_tokens = []
        for i, token in enumerate(example.doc_tokens):
            try:
                original_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_original_index.append(i)
                    all_doc_tokens.append(sub_token)
            except Exception as e:
                print(e)
                pass

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = original_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = original_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.original_answer_text)

        _DocSpan = collections.namedtuple(
            "DocSpan", ["start", "length"])

        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_seq_length:
                length = max_seq_length
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            doc_tokens = []
            token_to_original_map = {}
            token_is_max_context = {}

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_original_map[len(doc_tokens)] = tok_to_original_index[split_token_index]
                is_max_context = check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(doc_tokens)] = is_max_context
                doc_tokens.append(all_doc_tokens[split_token_index])

            start_position = None
            end_position = None
            if is_training:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    continue
                else:
                    start_position = tok_start_position - doc_start
                    end_position = tok_end_position - doc_start

            features.append(InputFeatures(
                qid=qid,
                uuid=uuid,
                doc_span_index=doc_span_index,
                token_to_original_map=token_to_original_map,
                token_is_max_context=token_is_max_context,
                doc_tokens=doc_tokens,
                ques_tokens=query_tokens,
                start_position=start_position,
                end_position=end_position))

            qid += 1

    return features


def token_process(features, tokenizer, vocab_file):
    #Processing the tokens
    word_counter, unk_counter, char_counter = Counter(), Counter(), Counter()
    for feature in tqdm(features):
        doc_tokens = feature.doc_tokens
        ques_tokens = feature.ques_tokens
        for i, token in enumerate(doc_tokens):
            if token not in tokenizer.vocab:
                unk_counter[token] += 1
            else:
                word_counter[token] += 1
            for char in token:
                char_counter[char] += 1
        for token in ques_tokens:
            if token not in tokenizer.vocab:
                unk_counter[token] += 1
            else:
                word_counter[token] += 1
            for char in token:
                char_counter[char] += 1

    print('UNK / HIT :', len(unk_counter), '/', len(word_counter))
    print('CHAR num :', len(char_counter))

    # Filter out vocabs not in dictionary, and give embedding to the ones that are indictionary
    word_embedding = tokenizer.get_word_embedding(word_counter, vocab_file, size=int(2.2e6), vec_size=300)
    char_embedding = tokenizer.get_char_embedding(char_counter, vec_size=64)

    return word_embedding, char_embedding, tokenizer


def build_features(features, tokenizer, save_path, max_seq_length=384, max_query_length=64, char_limit=16, is_training=True):
    #Save the data to a pickle file
    def convert_token_to_id(vocab, token):
        for each in (token, token.lower(), token.capitalize(), token.upper()):
            if each in vocab:
                return vocab[each]
        return vocab['--OOV--']
    def convert_char_to_id(vocab, char):
        if char in vocab:
            return vocab[char]
        return vocab['--OOV--']
    context_idxss = []
    ques_idxss = []
    context_char_idxss = []
    ques_char_idxss = []
    y1s = []
    y2s = []
    qids = []
    for feature in tqdm(features):
        try:
            qids.append(feature.qid)
            context_idxs = np.zeros([max_seq_length], dtype=np.int32)
            context_char_idxs = np.zeros([max_seq_length, char_limit], dtype=np.int32)
            ques_idxs = np.zeros([max_query_length], dtype=np.int32)
            ques_char_idxs = np.zeros([max_query_length, char_limit], dtype=np.int32)
            y1 = np.zeros([max_seq_length], dtype=np.float32)
            y2 = np.zeros([max_seq_length], dtype=np.float32)
            for i, token in enumerate(feature.doc_tokens):
                context_idxs[i] = convert_token_to_id(tokenizer.vocab, token)
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    context_char_idxs[i, j] = convert_char_to_id(tokenizer.char_vocab, char)
            for i, token in enumerate(feature.ques_tokens):
                ques_idxs[i] = convert_token_to_id(tokenizer.vocab, token)
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    ques_char_idxs[i, j] = convert_char_to_id(tokenizer.char_vocab, char)
            if is_training:
                y1[feature.start_position], y2[feature.end_position] = 1.0, 1.0
            context_idxss.append(np.expand_dims(context_idxs, axis=0))
            ques_idxss.append(np.expand_dims(ques_idxs, axis=0))
            context_char_idxss.append(np.expand_dims(context_char_idxs, axis=0))
            ques_char_idxss.append(np.expand_dims(ques_char_idxs, axis=0))
            if is_training:
                y1s.append(np.expand_dims(y1, axis=0))
                y2s.append(np.expand_dims(y2, axis=0))
        except Exception as e:
            print(e)
            pass
    context_idxss = np.concatenate(context_idxss, axis=0)
    ques_idxss = np.concatenate(ques_idxss, axis=0)
    context_char_idxss = np.concatenate(context_char_idxss, axis=0)
    ques_char_idxss = np.concatenate(ques_char_idxss, axis=0)
    if is_training:
        y1s = np.concatenate(y1s, axis=0)
        y2s = np.concatenate(y2s, axis=0)
    qids = np.array(qids)
    meta = {'qid': qids,
            'context_id': context_idxss,
            'question_id': ques_idxss,
            'context_char_id': context_char_idxss,
            'question_char_id': ques_char_idxss,
            'y_start': y1s,
            'y_end': y2s}
    print('save to', save_path, len(qids), 'features')
    with open(save_path, 'wb') as f:
        pickle.dump(meta, f)


if __name__ == '__main__':

    # Load tokenizer
    tokenizer = tokenization.FullTokenizer(vocab_file='original_data/glove.840B.300d.txt', do_lower_case=False)
    train_examples = read_squad_examples(input_file='original_data/train-v1.1.json', is_training=True)
    dev_examples = read_squad_examples(input_file='original_data/dev-v1.1.json', is_training=False)

    

    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_length=400, max_query_length=50,is_training=True)
    dev_features = convert_examples_to_features(dev_examples, tokenizer, max_seq_length=400, max_query_length=50, is_training=False)



    total_features = []
    total_features.extend(train_features)
    total_features.extend(dev_features)
    word_embedding, char_embedding, tokenizer = token_process(total_features, tokenizer, 'original_data/glove.840B.300d.txt')

    
    print(word_embedding.shape)
    print(len(tokenizer.vocab))
    print(char_embedding.shape)
    print(len(tokenizer.char_vocab))

    preprocessDatasetPath = Path('./dataset/')
    if not preprocessDatasetPath.is_dir():
        preprocessDatasetPath.mkdir()

    np.save('./dataset/word_emb_mat.npy', word_embedding)
    np.save('./dataset/char_emb_mat.npy', char_embedding)
    
    with open('./dataset/dev_examples.pkl', 'wb') as p:
        pickle.dump(dev_examples, p)
    with open('./dataset/dev_features.pkl', 'wb') as p:
        pickle.dump(dev_features, p)


    build_features(train_features, tokenizer, './dataset/train_set.pkl', is_training=True)
    build_features(dev_features, tokenizer, './dataset/dev_set.pkl', is_training=False)
