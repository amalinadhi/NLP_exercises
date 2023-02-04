import numpy as np
import pandas as pd


def readData(filename):
    # Read data
    df = pd.read_csv(filename)

    # Convert into list
    df = list(df["Data"])

    return df

def extractUniqueWord(sentence):
    # Normalize
    sentence = sentence.lower()

    # Obtain word list
    #word_list = sentence.split(" ")   # if you training with non berkeley data
    word_list = sentence.split(" ")[1:]

    return word_list            

def extractWordCorpus(list_corpus):
    # All word list
    corpus_word_list = []
    for sentence in list_corpus:
        # Extract word in each sentence
        word_list = extractUniqueWord(sentence = sentence)

        # Add word_list to corpus_word_list
        corpus_word_list += word_list

    # Remove unused elements
    # Use these depend on the case
    corpus_word_list = [x for x in corpus_word_list if "." not in x]
    corpus_word_list = [x for x in corpus_word_list if "<" not in x]
    corpus_word_list = [x for x in corpus_word_list if "[" not in x]
    corpus_word_list = [x for x in corpus_word_list if "_" not in x]
    corpus_word_list = [x for x in corpus_word_list if "-" not in x]
    
    # Remove duplicate words
    corpus_word_list, corpus_word_count = np.unique(corpus_word_list,
                                                    return_counts = True)
    corpus_word_list.sort()

    return corpus_word_list, corpus_word_count

def extractBigramMatrix(corpus_word_list, list_corpus):
    n = len(corpus_word_list)
    
    bigram_matrix = np.zeros((n, n))

    # Looping in all sentences
    for sentence in list_corpus:
        # Extract word list
        word_list = extractUniqueWord(sentence = sentence)

        # Loop for each first & next word
        for i, first_word in enumerate(corpus_word_list):
            for j, next_word in enumerate(corpus_word_list):
                # Iterate over word
                for k in range(len(word_list)-1):
                    condition_1 = word_list[k] == first_word
                    condition_2 = word_list[k+1] == next_word
                    
                    # If condition 1 & 2 achieved, then
                    if condition_1 and condition_2:
                        bigram_matrix[j, i] += 1

    return bigram_matrix

def trainBigram(list_corpus):
    # Extract corpus word list    
    corpus_word_list, corpus_word_count = extractWordCorpus(list_corpus = list_corpus)
    
    # Extract bigram matrix
    bigram_matrix = extractBigramMatrix(corpus_word_list = corpus_word_list,
                                        list_corpus = list_corpus)

    # Create bigram probability matrix
    bigram_matrix_prob = bigram_matrix/corpus_word_count

    # Create in pandas Dataframe
    bigram_matrix_prob = pd.DataFrame(data = bigram_matrix_prob,
                                      columns = corpus_word_list,
                                      index = corpus_word_list)

    return bigram_matrix_prob

def predict(test, bigram_matrix_prob):
    # normalize test
    test = test.lower()

    # split test
    test = test.split(" ")
    
    total_proba = 1.
    for i in range(len(test)-1):
        bigram = test[i:i+2]
        bigram_proba = bigram_matrix_prob[bigram[0]][bigram[1]]
        total_proba *= bigram_proba

    return total_proba



if __name__ == "__main__":
    # Read corpus
    FILENAME = "dataset/berkeley_restaurant_transcripts_small.csv"
    list_corpus = readData(filename = FILENAME)

    # Train bigram
    bigram_matrix_prob = trainBigram(list_corpus = list_corpus)

    # Predict
    TEST = "can you give me"
    test_proba = predict(test = TEST,
                         bigram_matrix_prob = bigram_matrix_prob)
    
    print(f"Sentence                : {TEST}")
    print(f"Probability to occur    : {test_proba:.4f}")

    
