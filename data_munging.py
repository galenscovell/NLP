"""
DATA MUNGING
=============
Basic data munging to produce cleaned data sets for other scripts.

@author GalenS <galen.scovell@gmail.com>
"""


import string
import nltk
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.stem.porter import PorterStemmer


punc = string.punctuation + '\n\r\t'
stop = stopwords.words('english')

bigram_measures = nltk.collocations.BigramAssocMeasures()
p_stemmer = PorterStemmer()


def stem(words):
    stemmed_words = [p_stemmer.stem(word) for word in words]
    return stemmed_words


def bigram(words, n):
    finder = BigramCollocationFinder.from_words(words)
    raw_bigrams = finder.nbest(bigram_measures.pmi, n)
    bigrams = ['_'.join(bigram) for bigram in raw_bigrams]
    return bigrams


def clean_data(file_name):
    cleaned = []
    with open(file_name, 'r') as f:
        for line in f:
            result = []
            for p in punc:
                line = line.replace(p, '')
            words = line.lower().rstrip().split(' ')
            for word in words:
                if word not in stop and not word.isdigit():
                    result.append(word)
            stemmed = stem(result)
            result = bigram(stemmed, 10)
            cleaned.append(' '.join(result))
    return cleaned



def save_data(file_name, contents):
    with open(file_name, 'w') as f:
        for c in contents:
            f.write(c + '\n')



if __name__ == '__main__':
    contents = clean_data('raw_dataset.txt')
    save_data('clean_dataset.txt', contents)
