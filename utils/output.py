import collections
import json
import math
import six
from tqdm import tqdm
from utils import tokenization

# This design was inspired from erfmca's implemntation that was mentioned in the report and also in train.py
def produce_final_prediction(predicted_text, original_text, lower_case, logger=None):
    """Project the tokenized prediction back to the original text."""

    # Sometimes the predicted text is like "john doe", while the ground truth is
    # "John Doe've", we do not want it to be in lower case and also not contain any 
    # extra content. So these functionsa im to achieve the target

    def clear_spaces(text):
        nchars = []
        nt_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            nt_map[len(nchars)] = i
            nchars.append(c)
        ntext = "".join(nchars)
        return (ntext, nt_map)

    # If "predicted_text" and "original_text" are of the same lengths, then they are likely aligned
    tokenizer = tokenization.BasicTokenizer(lower_case=lower_case)

    ttext = " ".join(tokenizer.tokenize(original_text))

    start_position = ttext.find(predicted_text)
    if start_position == -1:
        return original_text
    end_position = start_position + len(predicted_text) - 1

    (original_ntext, original_nts_map) = clear_spaces(original_text)
    (token_ntext, token_nts_map) = clear_spaces(ttext)

    if len(original_ntext) != len(token_ntext):
        return original_text

    # This part is doing character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(token_nts_map):
        tok_s_to_ns_map[tok_index] = i

    original_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in original_nts_map:
            original_start_position = original_nts_map[ns_start_position]

    if original_start_position is None:
        logger.info("Couldn't map start position")
        return original_text

    original_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in original_nts_map:
            original_end_position = original_nts_map[ns_end_position]

    if original_end_position is None:
        logger.info("Couldn't map end position")
        return original_text

    output_text = original_text[original_start_position:(original_end_position + 1)]
    return output_text


def write_predictions(examples, features, results, n_best_size,
                      max_answer_length, lower_case, output_prediction_file, output_nbest_file):
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    for feature in features:
        example_index_to_features[feature.uuid].append(feature)

    unique_id_to_result = {}
    for result in results:
        unique_id_to_result[result.qid] = result

    preliminary_prediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for example in tqdm(examples):
        features = example_index_to_features[example.uuid]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.qid]
            start_indexes = get_best_logits(result.start_logits, n_best_size)
            end_indexes = get_best_logits(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of
            # irrelevant
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.doc_tokens):
                        continue
                    if end_index >= len(feature.doc_tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        preliminary_prediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            tok_tokens = feature.doc_tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            ttext = " ".join(tok_tokens)

            # Clean whitespace
            ttext = ttext.strip()
            ttext = " ".join(ttext.split())
            original_text = " ".join(orig_tokens)

            final_text = produce_final_prediction(ttext, original_text, lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))


        #No valid prediction means that this 
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = float(entry.start_logit)
            output["end_logit"] = float(entry.end_logit)
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        nbest_json = sorted(nbest_json, key=lambda x: x['probability'], reverse=True)
        all_predictions[example.uuid] = nbest_json[0]['text']
        all_nbest_json[example.uuid] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


def get_best_logits(logits, n_best_size):
    # Get the n best logits from logits
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def softmax(scores):
    # This was a function that calculates softmax
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
