from time import perf_counter

from nltk import NaiveBayesClassifier

from typing import List, Dict, Tuple

from src.data_loaders import Review, load_amazon_train_test
from collections import Counter

from src.utils import stopwords_english, tokens, review_text_stats, aggregate_sentiment_results, \
    print_information_table_markdown


def determine_most_common_words(data: List[Review], n=2000, stopwords=None) -> Dict:
    if stopwords is None:
        stopwords = stopwords_english()
    word_counts = Counter()
    for review in data:
        ts = tokens(review.content, stopwords)
        word_counts.update(Counter(ts))
    return dict(word_counts.most_common(n))


def convert_most_common_words_to_features(most_common_words: Dict) -> Dict:
    return {common_word: False for common_word in most_common_words}


def convert_reviews_to_features(reviews: List[Review], feature_vector: Dict, stopwords=None) -> List[Tuple]:
    if stopwords is None:
        stopwords = stopwords_english()
    results = []
    for review in reviews:
        sentiment = review.sentiment
        content = review.content
        cur_features = feature_vector.copy()
        for token in tokens(content, stopwords):
            if token in cur_features:
                cur_features[token] = True
        results.append((cur_features, sentiment, review))
    return results


def train_bernoulli_classifier(reviews: List[Review], feature_vector: Dict) -> NaiveBayesClassifier:
    training_data = [(feats, label) for feats, label, _ in convert_reviews_to_features(reviews, feature_vector)]
    return NaiveBayesClassifier.train(training_data)


def test_on_amazon_reviews(train_size=2000, test_size=1000, feat_size=2000):
    train, test = load_amazon_train_test(train_size, test_size)
    run_test_on(train, test, 'Amazon', feat_size)


def run_test_on(train: List[Review], test: List[Review], review_source: str, feat_size=2000):
    most_common_words = determine_most_common_words(train + test, feat_size)
    feats = convert_most_common_words_to_features(most_common_words)

    test = convert_reviews_to_features(test, feats)
    classifier = train_bernoulli_classifier(train, feats)

    correctly_predicted = 0

    start = perf_counter()
    predictions = []
    for feat_vector, sentiment, review in test:
        text_length, common_words = review_text_stats(review)
        cur_sentiment = classifier.classify(feat_vector)
        if cur_sentiment == sentiment:
            correctly_predicted += 1
        prediction = [sentiment.lower(), cur_sentiment, text_length, common_words]
        predictions.append(prediction)
    end = perf_counter() - start

    predicted_sentiments = aggregate_sentiment_results(predictions)

    print(f"Predictions finished -- elapsed time: {end:3f} seconds")
    print("")

    # print information table as markdown
    print_information_table_markdown(review_source, len(train), len(test), feat_size, 0, predictions)
    print("")

    # print overall_correct_results
    print("# overall correctly predicted results")
    print(f"correct = ({correctly_predicted}, {len(test)})")

    # print results_by_sentiments
    # (d = [<label>, <predicted>, <correct>])
    print("# record format: (<sentiment>, [<label>, <predicted>, <correct>])")
    res = [(label, list(d.values())) for label, d in list(predicted_sentiments.items())]
    print(f"{review_source.lower()}_results = {res}")


def run():
    train_size = 20_000
    test_size = 3000
    feat_size = 3000
    test_on_amazon_reviews(train_size, test_size, feat_size)


if __name__ == '__main__':
    run()
