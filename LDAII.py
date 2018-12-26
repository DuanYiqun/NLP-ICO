import numpy as np
import pandas as pd
import re, nltk,  gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools

import matplotlib.pyplot as plt


import pandas as pd
import numpy as np

import nltk

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# preprocessing imports
from sklearn.preprocessing import LabelEncoder

# model imports
from gensim.models.ldamulticore import LdaMulticore

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# hpyerparameter training imports
from sklearn.model_selection import GridSearchCV

# visualization imports
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import io

sns.set()

# Settting input File Path
train_data = pd.read_csv("/Users/jinfeng/Documents/Test/test.csv")


# Measure the length of content
# do not use split(' ') which results in strange outcome to take the place of split()
content_length = np.array(list(map(len, train_data.text.str.split())))

print("The average number of words in a whitepaper is : {}.".format(np.mean(content_length)))
print("The min number of words in a whitepaper is : {}.".format(min(content_length)))
print("The max number of words in a whitepaper is : {}.".format(max(content_length)))


# Feature Creation
our_special_word = 'qwerty'

def remove_ascii_words(df):
    """
    remmove non-ascii characters from the 'texts' column in df.
    It returns the words containing non-ascii characters
    """
    non_ascii_words = []
    for i in range(len(df)):
        for word in df.loc[i, 'text'].split(' '):
            if any([ord(character) >= 128 for character in word]):
                non_ascii_words.append(word)
                # The word which ASCII is larger than 128 will be replaced with "qwert"
                df.loc[i, 'text'] = df.loc[i, 'text'].replace(word, our_special_word)

    return non_ascii_words

non_ascii_words = remove_ascii_words(train_data)
# print("Replace {} words with characters with an ordinal >= 128 in the train data.".format(len(non_ascii_words)))



def get_good_tokens(sentence):
    "Remove punctation and space"
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z?]+', '', token), sentence))
    # it is a unknown function here, I do not know what it does work for
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation




# Data preprocessing steps for LDA
def lda_get_good_tokens(df):
    df['text'] = df.text.str.lower()
    df['tokenized_text'] = list(map(nltk.word_tokenize, df.text))
    df['tokenized_text'] = list(map(get_good_tokens, df.tokenized_text))

lda_get_good_tokens(train_data)


def remove_stopwords(df):
    """Removes stopwords based on a known set of
    stopwords available in the nltk package. In addition, we
    include our made up word in here"""
    stopwords = nltk.corpus.stopwords.words('english')

    stopwords.append(our_special_word)
    df['stopwords_removed'] = list(map(lambda doc:
                                       [word for word in doc
                                        if word not in stopwords],
                                       df['tokenized_text']))


remove_stopwords(train_data)


# 2.3 Stemming
# including lemmatization and stemming
# Difference between lemmatization and stemming
# https://www.quora.com/What-is-difference-between-stemming-and-lemmatization

def stem_words(df):
    lemm = nltk.stem.WordNetLemmatizer()
    df['lemmatized_text'] = list(map(lambda sentence:
                                     list(map(lemm.lemmatize, sentence)),
                                     df.stopwords_removed))
    p_stemmer = nltk.stem.porter.PorterStemmer()
    df['stemmed_text'] = list(map(lambda sentence:
                                  list(map(p_stemmer.stem, sentence)),
                                  df.lemmatized_text))


stem_words(train_data)


# 2.4 Vectorize words
dictionary = Dictionary(documents=train_data.lemmatized_text.values)
print("Found {} words".format(len(dictionary.values())))

dictionary.filter_extremes(no_above=0.95, no_below=1.2)
print("Left with {} words.".format(len(dictionary.values())))

dictionary.compactify() # Reindexes the remaining words after filtering
print("Left with {} words.".format(len(dictionary.values())))

# Make a BOW(Bag-of-Words Model) for every document
def document_to_bow(df):
    df['bow'] = list(map(lambda doc:
                         dictionary.doc2bow(doc),
                         df.lemmatized_text))


document_to_bow(train_data)

def lda_preprocessing(df):
    """All the preprocessing steps for LDA are combined in this function.
    All mutations are done on the dataframe itself.
    So this function returns nothing"""
    lda_get_good_tokens(df)
    remove_stopwords(df)
    stem_words(df)
    document_to_bow(df)

corpus = train_data.bow


# # Build LDA model
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=dictionary,
#                                            num_topics=20,
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha='auto',
#                                            per_word_topics=True)
#
#
# # Print the Keyword in the 10 topics
# pprint(lda_model.print_topics())
# doc_lda = lda_model[corpus]

mallet_path = "/Users/jinfeng/Downloads/mallet-2.0.8/bin/mallet"


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=train_data.lemmatized_text, start=20, limit=200, step=5)

# Show graph
limit=200; start=20; step=5;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()