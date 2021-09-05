from random import shuffle

import pandas as pd

from dataclasses import dataclass
from typing import List, Tuple, Dict

# https://www.kaggle.com/arhamrumi/amazon-product-reviews
AMAZON_REVIEWS = ['data/amazon_misc_products_reviews.csv']

POSITIVE_WORDS = 'data/positive-words.txt'
NEGATIVE_WORDS = 'data/negative-words.txt'


POS = 'positive'
NEG = 'negative'


@dataclass
class Review:
    rev_id: int
    content: str
    sentiment: str


def load_pos_neg_words() -> Tuple:
    with open(POSITIVE_WORDS, 'r') as pos_file, open(NEGATIVE_WORDS, 'r') as neg_file:
        pos = [w.strip() for w in pos_file.readlines() if w.strip()]
        neg = [w.strip() for w in neg_file.readlines() if w.strip()]
        return pos, neg


def load_pos_neg_lookup() -> Dict:
    pos, neg = load_pos_neg_words()
    pos = {word: POS for word in pos}
    neg = {word: NEG for word in neg}
    return {**pos, **neg}


def load_reviews(data_files: List[str]) -> List[Review]:
    results = []
    for rev_file in data_files:
        reviews = pd.read_csv(rev_file, encoding='utf-8')
        selection = reviews[['title', 'text', 'rating']]
        for i, review in enumerate(selection.values.tolist()):
            content = f"{review[0]}\n{review[1]}"
            rating = review[2]
            if rating == 3:
                continue
            sentiment = NEG if rating < 3 else POS
            results.append(Review(i, content, sentiment))
    return results


def split_train_test(reviews: List[Review], sentiment: str, train_size: int, test_size: int) -> Tuple:
    reviews_by_sentiment = [review for review in reviews if review.sentiment == sentiment]
    shuffle(reviews_by_sentiment)
    train = reviews_by_sentiment[:train_size]
    test = reviews_by_sentiment[train_size:train_size+test_size]
    return train, test


def load_train_test_split(data_files, train_size, test_size) -> Tuple:
    train_res = []
    test_res = []
    reviews = load_reviews(data_files)

    train, test = split_train_test(reviews, POS, train_size, test_size)
    train_res += train
    test_res += test

    train, test = split_train_test(reviews, NEG, train_size, test_size)
    train_res += train
    test_res += test

    return train_res, test_res


def load_amazon_train_test(train_size=2000, test_size=1000):
    return load_train_test_split(AMAZON_REVIEWS, train_size, test_size)


def run():
    lookup = load_pos_neg_lookup()
    print("")


if __name__ == '__main__':
    run()
