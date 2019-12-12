import classifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

test_file = "./evaluation_examples.csv"
test_data = classifier.load_data(test_file)
test_text = test_data[0]

prediction = classifier.classify(test_text, './polarity_classifier.sav')

test_text_pos = []
test_text_neg = []

for pol in range(len(prediction)):
    if prediction[pol] == 3:
        test_text_pos.append(test_text[pol])
    if prediction[pol] == 4:
        test_text_neg.append(test_text[pol])

def create_tf_idf_matrix(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    tf_idf_matrix = vectorizer.fit_transform(documents)
    scores = zip(vectorizer.get_feature_names(),
                     np.asarray(tf_idf_matrix.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    for item in sorted_scores:
        print("{0:50} Score: {1}".format(item[0], item[1]))

    return sorted_scores

#create_tf_idf_matrix(test_text_pos)
#create_tf_idf_matrix(test_text_neg)

sorted_scores = create_tf_idf_matrix(test_text_pos)
#sorted_scores = [i[0] for i in sorted_scores if nlp(i[0])[0].pos_ == 'VERB']
#print(sorted_scores)