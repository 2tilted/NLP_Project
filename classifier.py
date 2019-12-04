import pandas as pd
from sklearn import metrics
import pickle



def load_data(test_file):
    test_data = pd.read_csv(test_file, sep=",", header=None)
    return test_data


def classify(test_data, model_name):
    vectorizer_name = './vectorizer.sav'
    vectorizer = pickle.load(open(vectorizer_name, 'rb'))
    classifier = pickle.load(open(model_name, 'rb'))
    test_tfidf_matrix = vectorizer.transform(test_data)
    prediction = classifier.predict(test_tfidf_matrix)
    return prediction


def calculate_domain(test_file):
    test_data = load_data(test_file)
    test_text, labels_test = test_data[0], test_data[1]
    model_name = './domain_classifier.sav'
    prediction = classify(test_text, model_name)
    print("Here is the accuracy for domain:", metrics.accuracy_score(prediction, labels_test) * 100, '%')


def calculate_polarity(test_file):
    test_data = load_data(test_file)
    test_text, labels_test = test_data[0], test_data[2]
    model_name = './polarity_classifier.sav'
    prediction = classify(test_text, model_name)
    print("Here is the accuracy for polarity:", metrics.accuracy_score(prediction, labels_test) * 100, '%')


if __name__ == '__main__':

    test = "./evaluation_examples.csv"
    calculate_domain(test)
    calculate_polarity(test)
