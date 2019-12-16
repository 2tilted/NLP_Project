import classifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

test_file = "./evaluation_examples.csv"
test_data = classifier.load_data(test_file)
test_text = test_data[0]

prediction = classifier.classify(test_text, './polarity_classifier.sav')

test_text_pos = []
test_text_neg = []

"""filter between positive and negative"""
for pol in range(len(prediction)):
    if prediction[pol] == 3:
        test_text_pos.append(test_text[pol])
    if prediction[pol] == 4:
        test_text_neg.append(test_text[pol])

def create_tf_idf_matrix(documents):
    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    tf_idf_matrix = vectorizer.fit_transform(documents)
    scores = zip(vectorizer.get_feature_names(),
                     np.asarray(tf_idf_matrix.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return sorted_scores

"""create tf idf matrices and filter out the non-adjectives"""
sorted_scores_pos = create_tf_idf_matrix(test_text_pos)
#sorted_scores_pos = [i[0] for i in sorted_scores_pos]
sorted_scores_pos = [i[0] for i in sorted_scores_pos if nlp(i[0])[0].pos_ == 'ADJ']

sorted_scores_neg = create_tf_idf_matrix(test_text_neg)
#sorted_scores_neg = [i[0] for i in sorted_scores_neg]
sorted_scores_neg = [i[0] for i in sorted_scores_neg if nlp(i[0])[0].pos_ == 'ADJ']

"""remove identical adjectives"""
for x in sorted_scores_pos[:]:
    if x in sorted_scores_neg:
        sorted_scores_neg.remove(x)
        sorted_scores_pos.remove(x)

#for ngram in sorted_scores_pos[:]:
 #   doc = nlp(ngram)
  #  for token in doc:
   #     if token.pos_ == 'NOUN' or token.pos_ == 'PROPN' or token.pos_ == 'NUM' or token.pos_ == 'SYM':
    #        sorted_scores_pos.remove(ngram)
     #       break

print(sorted_scores_neg)

"""create new csv file with modified text"""
df = pd.read_csv('./evaluation_examples.csv', sep=",", header=None)
for sentence in range(len(df[0])):
    for word in df[0][sentence].split():
        if word in sorted_scores_pos:
            s = df.at[sentence, 0]
            s = s.replace(word,sorted_scores_neg[0])
            print(s)
            df.at[sentence, 0] = s
        elif word in sorted_scores_neg:
            s = df.at[sentence, 0]
            s = s.replace(word,sorted_scores_pos[0])
            print(s)
            df.at[sentence, 0] = s
df.to_csv('copy_of_' + 'evaluation_examples.csv', index=False, header=None)