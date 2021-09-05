from statistics import mean
from typing import List, Tuple, Set

import nltk

from nltk.tokenize import TreebankWordTokenizer

from re import sub

from data_loaders import Review, POS, NEG

COMMON = 'data/en_most_common.txt'
COMMON_POL = 'data/en_common_words_polarity.csv'

TKN = nltk.data.load('tokenizers/punkt/english.pickle')


def stopwords_english(excludes: List = None) -> Set:
    if not excludes:
        excludes = ['not']
    return {sw for sw in nltk.corpus.stopwords.words('english') if sw not in excludes}


def common_words_english() -> Set:
    with open(COMMON, 'r', encoding='utf-8') as f:
        words = f.readlines()
        stopwords = stopwords_english()
        return {word.strip() for word in words if word.strip().lower() not in stopwords}


def sentences(post: str) -> List[str]:
    return TKN.tokenize(post)


def tokens(text: str, stopwords=None, lowercase=True) -> List[str]:
    if stopwords is None:
        stopwords = stopwords_english(['not'])
    # replace newline with whitespace
    text = text.replace('\n', ' ')

    # insert whitespace between [A-z]<punct>[A-z]
    text = sub(r"([A-z])([^A-z'])([A-z])", r"\1\2 \3", text)

    # insert whitespace between [a-z][A-Z] -> caution: would separate words such as "iPhone", too!
    text = sub(r"([a-z])([A-Z])", r"\1 \2", text)

    res = [t for t in TreebankWordTokenizer().tokenize(text) if t.lower() not in stopwords]
    res = [t.replace("n't", "not") for t in res]
    res = [t for t in res if t.isalpha()]

    if lowercase:
        return [t.lower() for t in res]
    return res


def review_text_stats(review: Review, stopwords=None, common_words=None) -> Tuple[int, float]:
    if not stopwords:
        stopwords = stopwords_english()
    if not common_words:
        common_words = common_words_english()
    tkns = [t.strip() for t in tokens(review.content, stopwords) if t.strip().isalpha()]
    if not tkns:
        return 0, 0
    common_words = [t for t in tkns if t.lower() in common_words]
    text_length = len(tkns)
    common_words_share = len(common_words) / len(tkns)
    return text_length, common_words_share


def aggregate_sentiment_results(prediction_results: List):
    sentiment_predicted = {
        POS: {'label': 0, 'predicted': 0, 'correct': 0},
        NEG: {'label': 0, 'predicted': 0, 'correct': 0},
    }
    for cur_label, cur_pred, cur_tl, cur_cw in prediction_results:
        if cur_label not in sentiment_predicted or cur_pred not in sentiment_predicted:
            continue
        sentiment_predicted[cur_label]['label'] += 1
        sentiment_predicted[cur_pred]['predicted'] += 1
        if cur_label == cur_pred:
            sentiment_predicted[cur_label]['correct'] += 1
    return sentiment_predicted


def print_information_table_markdown(review_source: str, train_size: int, test_size: int, feat_size: int, epochs: int, predictions: List):
    text_lengths = [res[2] for res in predictions]
    common_word_shares = [res[3] for res in predictions]
    avg_text_len = int(mean(text_lengths))
    avg_cw_share = mean(common_word_shares) * 100
    text_len_min = min(text_lengths)
    text_len_max = max(text_lengths)

    md_table = []
    md_table.append(f"| {review_source.capitalize()} Reviews : Test Setup | |")
    md_table.append("| :--- | ---: |")
    md_table.append(f"| Training Sample Size | {train_size} |")
    md_table.append(f"| Test Sample Size | {test_size} |")
    if 0 < feat_size:
        md_table.append(f"| Feature vector size | {feat_size} |")
    if 0 < epochs:
        md_table.append(f"| Training epochs | {epochs} |")
    md_table.append(f"| **Information about {review_source.capitalize()} Reviews** | |")
    md_table.append(f"| Avg. text length (in tokens) | {avg_text_len:7} |")
    md_table.append(f"| Min text length (in tokens) | {text_len_min:7} |")
    md_table.append(f"| Max text length (in tokens) |  {text_len_max:7} |")
    md_table.append(f"| Avg. share of 3000 most common words | {avg_cw_share:.2f}% |")

    for line in md_table:
        print(line)
