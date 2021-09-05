from time import perf_counter
from typing import List

from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from data_loaders import Review, load_amazon_train_test
from utils import tokens, stopwords_english, review_text_stats, aggregate_sentiment_results, \
    print_information_table_markdown


def review2tagged_doc(review: Review, stopwords=None) -> TaggedDocument:
    if stopwords is None:
        stopwords = stopwords_english()
    return TaggedDocument(tokens(review.content, stopwords), [review.sentiment])


def reviews2tagged_docs(reviews: List[Review], stopwords=None) -> List:
    if stopwords is None:
        stopwords = stopwords_english()
    return [review2tagged_doc(review, stopwords) for review in reviews]


def train_doc2vec_model(reviews: List[Review], stopwords=None, vec_size=100, epochs=10):
    if stopwords is None:
        stopwords = stopwords_english(['not'])
    train = reviews2tagged_docs(reviews, stopwords)
    model = Doc2Vec(vector_size=vec_size, min_count=2, epochs=epochs)
    model.build_vocab(train)
    model.train(train, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def test_on_amazon_reviews(train_size=2000, test_size=1000):
    train, test = load_amazon_train_test(train_size, test_size)
    run_test_on(train, test, 'Amazon')


def run_test_on(train: List[Review], test: List[Review], review_source: str):
    epochs = 20
    feat_size = 300

    stopwords = stopwords_english(['not'])
    model = train_doc2vec_model(train, stopwords, feat_size, epochs)

    correctly_predicted = 0

    start = perf_counter()
    predictions = []
    for review in test:
        text_length, common_words = review_text_stats(review)
        sentiment = review.sentiment
        content = [token.lower() for token in tokens(review.content, stopwords)]
        content_vector = model.infer_vector(content)
        similarities = model.dv.most_similar([content_vector], topn=1)
        cur_sentiment, certainty = similarities[0]
        if cur_sentiment == sentiment:
            correctly_predicted += 1
        prediction_result = [sentiment.lower(), cur_sentiment, text_length, common_words]
        predictions.append(prediction_result)
    end = perf_counter() - start

    sentiment_predicted = aggregate_sentiment_results(predictions)

    print(f"Predictions finished -- elapsed time: {end:3f} seconds")
    print("")

    # print information table as markdown
    print_information_table_markdown(review_source, len(train), len(test), feat_size, epochs, predictions)
    print("")

    # print overall_correct_results
    print("# overall correctly predicted results")
    print(f"correct = ({correctly_predicted}, {len(test)})")

    # print results_by_sentiments
    # (d = [<label>, <predicted>, <correct>])
    print("# record format: (<sentiment>, [<label>, <predicted>, <correct>])")
    res = [(label, list(d.values())) for label, d in list(sentiment_predicted.items())]
    print(f"{review_source.lower()}_results = {res}")
    print("")


def run():
    train_size = 20_000
    test_size = 3000
    test_on_amazon_reviews(train_size, test_size)


if __name__ == '__main__':
    run()
