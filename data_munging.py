"""
DATA MUNGING
=============
Basic data munging to produce cleaned data sets for other scripts.

@author GalenS <galen.scovell@gmail.com>
"""


import string
from nltk.corpus import stopwords


punc = string.punctuation + '\n\r\t'
stop = stopwords.words('english')


def clean_data(file_name):
    cleaned = []

    with open(file_name, 'r') as f:
        for line in f:
            cleaned_words = []
            for p in punc:
                line = line.replace(p, '')
            words = line.lower().rstrip().split(' ')
            for word in words:
                if word not in stop and not word.isdigit():
                    cleaned_words.append(word)
            cleaned.append(' '.join(cleaned_words))

    return cleaned



def save_data(file_name, contents):
    with open(file_name, 'w') as f:
        for c in contents:
            f.write(c + '\n')



if __name__ == '__main__':
    contents = clean_data('raw_dataset.txt')
    save_data('clean_dataset.txt', contents)
