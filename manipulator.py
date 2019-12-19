import classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.sparsefuncs import csc_median_axis_0
import numpy as np
import pandas as pd
import spacy
import re
import pickle
import itertools

nlp = spacy.load("en_core_web_sm")

test_file = "./evaluation_examples.csv"
test_data = classifier.load_data(test_file)
test_text = test_data[0]

polarity_prediction = classifier.classify(test_text, './polarity_classifier.sav')
domain_prediction = classifier.classify(test_text, './domain_classifier.sav')

test_text_0_3 = []
test_text_1_3 = []
test_text_0_4 = []
test_text_1_4 = []

for idx in range(len(polarity_prediction)):
    if domain_prediction[idx] == 0 and polarity_prediction[idx] == 3:
        test_text_0_3.append(test_text[idx])
    elif domain_prediction[idx] == 1 and polarity_prediction[idx] == 3:
        test_text_1_3.append(test_text[idx])
    elif domain_prediction[idx] == 0 and polarity_prediction[idx] == 4:
        test_text_0_4.append(test_text[idx])
    else:
        test_text_1_4.append(test_text[idx])


vectorizer = TfidfVectorizer(ngram_range=(1,2))
def create_tf_idf_matrix(documents):
    tf_idf_matrix = vectorizer.fit_transform(documents)
    scores = zip(vectorizer.get_feature_names(),
                     np.asarray(tf_idf_matrix.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return sorted_scores

sorted_scores_0_3 = create_tf_idf_matrix(test_text_0_3)
sorted_scores_1_3 = create_tf_idf_matrix(test_text_1_3)
sorted_scores_0_4 = create_tf_idf_matrix(test_text_0_4)
sorted_scores_1_4 = create_tf_idf_matrix(test_text_1_4)

def filter_polarity(pos_list, neg_list):
    for word in neg_list[:]:
        tmp = [item for item in pos_list if item[0] == word[0]]
        if len(tmp) == 1:
            q = tmp[0][1]/word[1]

            if q < .2: #the ngram has negative polarity
                pos_list.remove(tmp[0])
            elif q > 5: #the ngram has positive polarity
                neg_list.remove(word)
            else: #the ngram is neutral
                pos_list.remove(tmp[0])
                neg_list.remove(word)

filter_polarity(sorted_scores_0_3, sorted_scores_0_4)
filter_polarity(sorted_scores_1_3, sorted_scores_1_4)

#for x in sorted_scores_pos[:]:
#    if x in sorted_scores_neg:
#        sorted_scores_neg.remove(x)
#        sorted_scores_pos.remove(x)

#for ngram in sorted_scores_pos[:]:
 #   doc = nlp(ngram)
  #  for token in doc:
   #     if token.pos_ == 'NOUN' or token.pos_ == 'PROPN' or token.pos_ == 'NUM' or token.pos_ == 'SYM': py
    #        sorted_scores_pos.remove(ngram)
     #       break

print("electronic positives:")
print(sorted_scores_0_3)
print("\nkitchen positives:")
print(sorted_scores_1_3)
print("\nelectronic negatives:")
print(sorted_scores_0_4)
print("\nkitchen negatives:")
print(sorted_scores_1_4)

sorted_scores_0_3 = [i[0] for i in sorted_scores_0_3]
sorted_scores_1_3 = [i[0] for i in sorted_scores_1_3]
sorted_scores_0_4 = [i[0] for i in sorted_scores_0_4]
sorted_scores_1_4 = [i[0] for i in sorted_scores_1_4]


vectorizer = TfidfVectorizer()
analyzer = vectorizer.build_analyzer()
preprocessor = vectorizer.build_preprocessor()

replacement_dict = {i : sorted_scores_0_4[0] for i in sorted_scores_0_3}
replacement_dict_1 = {i : sorted_scores_0_3[0] for i in sorted_scores_0_4}
replacement_dict_2 = {i : sorted_scores_1_4[0] for i in sorted_scores_1_3}
replacement_dict_3 = {i : sorted_scores_1_3[0] for i in sorted_scores_1_4}

replacement_dict.update(replacement_dict_1)
replacement_dict.update(replacement_dict_2)
replacement_dict.update(replacement_dict_3)

def multi_replace_regex(string, replacements, ignore_case=False):
    rep_sorted = sorted(replacements, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)
    pattern = re.compile("|".join(rep_escaped))
    return pattern.sub(lambda match: replacements[match.group(0)], string)

df = pd.read_csv('./evaluation_examples.csv', sep=",", header=None)

for sentence in range(len(df[0])):
    s = df.at[sentence, 0]
    s = multi_replace_regex(s, replacement_dict)
    df.at[sentence, 0] = s
"""
for sentence in range(len(df[0])):
    for word in df[0][sentence].split():
        if word in sorted_scores_0_3:
            s = df.at[sentence, 0]
            s = s.replace(word, sorted_scores_0_4[0])
            print(s)
            df.at[sentence, 0] = s
        elif word in sorted_scores_0_4:
            s = df.at[sentence, 0]
            s = s.replace(word, sorted_scores_0_3[0])
            print(s)
            df.at[sentence, 0] = s

        elif word in sorted_scores_1_3:
            s = df.at[sentence_idx, 0]
            s = s.replace(word, sorted_scores_1_4[0])
            if len(word.split()) > 1:
                print("BIGRAM \n")
            #print(s)
            df.at[sentence_idx, 0] = s
        elif word in sorted_scores_1_4:
            s = df.at[sentence_idx, 0]
            s = s.replace(word, sorted_scores_1_3[0])
            if len(word.split()) > 1:
                print("BIGRAM \n")
            #print(s)
            df.at[sentence_idx, 0] = s
        #else:
            #print("nothing found")
"""

df.to_csv('copy_of_' + 'evaluation_examples.csv', index=False, header=None)