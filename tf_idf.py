import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold


def readData(filename):
    """
    Read data
    """
    # Read data
    data = pd.read_csv(filename)

    # Convert data to list
    data = list(data["Data"])

    return data

def readData2(filename):
    # read data
    data = pd.read_csv(filename)

    # convert to list
    data = list(data["Data"])

    # Clean data
    cleaned_data = []
    for data_i in data:
        # 1. Split data
        data_i_split = data_i.split(" ")[1:]

        # 2. Remove unnecessary things
        char_to_remove = ["!", ".", "(", ")", "*", "-", "<", ">",
                          "[", "]", "{", "}", "_", "?"]
        data_i_cleaned = removeWord(splitted_data = data_i_split,
                                    char_to_remove = char_to_remove)

        # 3. Join data
        data_i_cleaned = " ".join(data_i_cleaned)
        cleaned_data.append(data_i_cleaned)

    return cleaned_data

def removeWord(splitted_data, char_to_remove):
    for char in char_to_remove:
        splitted_data = [x for x in splitted_data if char not in x]

    return splitted_data
    
def normalizeSentence(sentence):
    """
    Normalize sentence function
    """
    # Lowering word in sentence
    sentence = sentence.lower()
    sentence = sentence.strip()

    return sentence

def extractWordCorpus(corpus_data):
    """
    Extract word corpus from corpus data.
    """
    # 1. Join the corpus data
    joined_corpus = " ".join(corpus_data)

    # 2. Normalize joined corpus
    normalized_corpus = normalizeSentence(sentence = joined_corpus)
    
    # 3. Extract unique word in a sentence
    word_corpus_raw = normalized_corpus.split(" ")
    word_corpus = np.unique(word_corpus_raw)[1:]

    return word_corpus

def computeTFIDF(corpus_data, word_corpus):
    """
    create TF matrix
    """
    # Create matrix
    n = len(word_corpus)
    m = len(corpus_data)

    word_count_matrix = pd.DataFrame(data = np.zeros((n, m)),
                                     columns = np.arange(1, m+1),
                                     index = word_corpus)
    
    # Calculate the word count matrix
    for i, corpus in enumerate(corpus_data):
        # Normalize the corpus
        corpus_normalized = normalizeSentence(sentence = corpus)

        # Split corpus into words
        corpus_normalized = corpus_normalized.split(" ")

        # Extract unique value & its counts
        word_list, word_count = np.unique(corpus_normalized,
                                          return_counts = True)

        for j, word in enumerate(word_list):
            if word in word_count_matrix.index:
                word_count_matrix[i+1].loc[word] += word_count[j]

    # Calculate the IDF matrix
    IDF_matrix = np.log10(m / (word_count_matrix > 0).sum(axis = 1))

    # Calculate the TF matrix
    TF_matrix = word_count_matrix#.apply(lambda x: np.log10(x+1))

    # Calculate the TF_IDF matrix
    TF_IDF_matrix = TF_matrix.mul(IDF_matrix, axis = 0)

    return TF_IDF_matrix

def findSimilarWord(test_word, n_word, tf_idf_matrix):
    words_sim = np.zeros(len(tf_idf_matrix.index))
    for i, word in enumerate(tf_idf_matrix.index):
        similarity = cosine_similarity(word_1 = tf_idf_matrix.loc[test_word],
                                       word_2 = tf_idf_matrix.loc[word])

        words_sim[i] = similarity
    
    sorted_idx = np.argsort(words_sim)[::-1]
    sorted_word_sim = words_sim[sorted_idx]
    sorted_word = list(tf_idf_matrix.index[sorted_idx[1:n_word+1]])

    print(f"Test word        : {test_word}")
    print(f"Most similar word:")
    print(sorted_word)

def findSimilarDoc(test_doc, n_doc, tf_idf_matrix, corpus_data):
    doc_sim = np.zeros(len(tf_idf_matrix.columns))
    for i, doc in enumerate(tf_idf_matrix.columns):
        similarity = cosine_similarity(word_1 = tf_idf_matrix[test_doc],
                                       word_2 = tf_idf_matrix[doc])

        doc_sim[i] = similarity
    
    sorted_idx = np.argsort(doc_sim)[::-1]
    sorted_doc_sim = doc_sim[sorted_idx]
    sorted_doc = np.array(corpus_data)[sorted_idx[1:n_doc+1]]

    print(f"Test doc        : {corpus_data[test_doc]}")
    print(f"Most similar doc:")
    print(sorted_doc)

def cosine_similarity(word_1, word_2):
    return np.dot(word_1, word_2)


if __name__ == "__main__":

    # 1. Read corpus data
    #FILENAME = "dataset/simple_corpus_03.csv"
    FILENAME = "dataset/berkeley_restaurant_transcripts.csv"
    corpus_data = readData2(filename = FILENAME)

    # 2. Create word corpus
    word_corpus = extractWordCorpus(corpus_data = corpus_data)
    print(word_corpus)

    # 3. Compute TF
    tf_idf_matrix = computeTFIDF(corpus_data = corpus_data,
                                 word_corpus = word_corpus)
    # save tf_idf_matrix
    #tf_idf_matrix.to_pickle("dataset/tf_idf/tf_idf_matrix.pickle")

    # load tf_idf_matrix
    #tf_idf_matrix = pd.read_pickle("dataset/tf_idf/tf_idf_matrix.pickle")
    #print(tf_idf_matrix)
    #print(tf_idf_matrix.index)

    # 5. Find the most similar words
    test_word = "food"
    n = 10
    findSimilarWord(test_word = test_word,
                    n_word = n,
                    tf_idf_matrix = tf_idf_matrix)

    # 6. Find the most similar documents
    print("")
    test_doc = 95
    n_doc = 10
    findSimilarDoc(test_doc = test_doc,
                   n_doc = n_doc,
                   tf_idf_matrix = tf_idf_matrix,
                   corpus_data = corpus_data)
    
