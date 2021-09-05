from random import shuffle
from time import perf_counter
from typing import List
from collections import Counter

from data_loaders import NEG, POS, Review, load_pos_neg_lookup, load_amazon_train_test
from utils import stopwords_english, tokens, review_text_stats, aggregate_sentiment_results, \
    print_information_table_markdown


def text_sentiment(text: str, pos_neg_lookup=None, stopwords=None) -> str:
    if pos_neg_lookup is None:
        pos_neg_lookup = load_pos_neg_lookup()
    if stopwords is None:
        stopwords = stopwords_english()
    sentiment_vector = [pos_neg_lookup.get(t, '') for t in tokens(text, stopwords)]
    sentiment_counts = Counter(sentiment_vector)
    sentiment = sentiment_counts.get(POS, 0) - sentiment_counts.get(NEG, 0)
    if sentiment == 0:
        return ''
    return POS if 0 < sentiment else NEG


def test_on_amazon_reviews(nrows=3000):
    _, test = load_amazon_train_test(0, nrows)
    shuffle(test)
    reviews = test[:nrows]
    run_test(reviews, 'Amazon')


def run_test(reviews: List[Review], review_source: str):
    num = len(reviews)
    stopwords = stopwords_english()
    sentiment_lookup = load_pos_neg_lookup()

    print(f"[Rule based] Start predicting sentiment for {num} {review_source} reviews")
    correctly_predicted = 0

    start = perf_counter()
    predictions = []
    for review in reviews[:num]:
        text_length, common_words = review_text_stats(review)
        sentiment = text_sentiment(review.content, sentiment_lookup, stopwords)
        if sentiment and sentiment == review.sentiment.lower():
            correctly_predicted += 1
        predition = [review.sentiment.lower(), sentiment, text_length, common_words]
        predictions.append(predition)
    end = perf_counter() - start

    predicted_sentiments = aggregate_sentiment_results(predictions)

    print(f"Predictions finished -- elapsed time: {end:3f} seconds")
    print("")

    # print information table as markdown
    print_information_table_markdown(review_source, 0, num, len(sentiment_lookup), 0, predictions)
    print("")

    # print overall_correct_results
    print("# overall correctly predicted results")
    print(f"correct = ({correctly_predicted}, {num})")

    # print results_by_sentiments
    # (d = [<label>, <predicted>, <correct>])
    print("# record format: (<sentiment>, [<label>, <predicted>, <correct>])")
    res = [(label, list(d.values())) for label, d in list(predicted_sentiments.items())]
    print(f"{review_source.lower()}_results = {res}")
    print("")


def run():
    test_on_amazon_reviews()


if __name__ == '__main__':
    run()
