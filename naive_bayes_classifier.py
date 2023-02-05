import numpy as np
import pandas as pd


def readData(filename):
    """
    Read sample data
    """
    df = pd.read_csv(filename)

    return df

def splitInputOutput(data, target):
    """
    split data to X and y
    """
    X = data.drop(columns = target)
    y = data[target]

    return X, y

def normalizeDoc(doc):
    """
    Normalize the doc
    """
    doc = doc.lower()

    return doc

def extractDocuments(y):
    """
    Calculate class proportion in whole documents
    """
    return y.value_counts(normalize = True)

def extractWordCountMatrix(X, y, word_list):
    """
    Generate word count matrix
    """
    # Extract class
    C = np.unique(y)
    n = len(word_list)
    m = len(C)
    
    # Create probability matrics
    word_matrix = pd.DataFrame(data = np.zeros((n, m)),
                               columns = C,
                               index = word_list)
    
    # Iterate over document class
    for c in C:
        # Extract doc
        current_doc_list = X["Data"].loc[y == c]

        # Extract current word list
        current_word_list = extractNonUniqueWord(docs = current_doc_list)
        
        # Fill the word matrix
        for word in current_word_list:
            word_matrix[c].loc[word] += 1 

    return word_matrix

def extractWordProbaMatrix(word_count_matrix, alpha):
    """
    Generate word probability matrix
    """
    # Find the number of word in class
    n_word_class = np.sum(word_count_matrix, 
                          axis = 0)
    n_word_total = len(word_count_matrix.index)
    

    # Calculate the word's conditional probability given class
    word_proba_matrix = word_count_matrix.copy()
    C = word_proba_matrix.columns
    for c in C:
        # Apply alpha
        word_proba_matrix[c] = word_count_matrix[c].apply(lambda x: x+alpha)

        # Calculate proba
        word_proba_matrix[c] = word_proba_matrix[c] / (n_word_class[c] + n_word_total)
    
    return word_proba_matrix

def extractNonUniqueWord(docs):
    """
    Return list of non-unique word
    """
    word_list_raw = []
    for doc in docs:
        doc = normalizeDoc(doc)
        doc = doc.split(" ")
        word_list_raw += doc

    return word_list_raw

def extractUniqueWord(docs):
    """
    Return list of unique word
    """
    word_list_raw = extractNonUniqueWord(docs = docs)
    word_list = np.unique(word_list_raw)
    
    return word_list
    
def trainNaiveBayes(data, alpha):
    """
    Train the Naive Bayes Classification
    """
    # Split input output
    X, y = splitInputOutput(data = data,
                            target = "Class")

    # Extract class proportion
    class_proportion = extractDocuments(y = y)
    log_prior = np.log(class_proportion)

    # Extract vocabulary
    word_list = extractUniqueWord(docs = X["Data"])

    # Create probability matrices for each class
    word_count_matrix = extractWordCountMatrix(X, y, word_list)
    word_proba_matrix = extractWordProbaMatrix(word_count_matrix = word_count_matrix,
                                               alpha = alpha)

    # Calculate the likelihood
    word_likelihood_matrix = np.log(word_proba_matrix)

    return log_prior, word_likelihood_matrix, word_list

def predictNaiveBayes(test, log_prior, word_likelihood_matrix):
    """
    Use to predict the doc setiment class
    """    
    # Extract word list
    test_word_list_raw = extractNonUniqueWord(docs = test)

    # Remove word that is not in the word list
    word_list = word_likelihood_matrix.index
    test_word_list = [word for word in test_word_list_raw if word in word_list]
    
    # Calculate the joint probability
    C = word_likelihood_matrix.columns.values
    class_proba = pd.DataFrame(data = log_prior.to_list(),
                               columns = ["proba"],
                               index = C)
    for c in C:
        for word in test_word_list:
            # Find the word proba
            word_proba = word_likelihood_matrix[c].loc[word]

            # Add the word proba to class proba
            class_proba.loc[c] += word_proba
       
    predicted_class = class_proba.idxmax().values
    predicted_log_proba = class_proba["proba"].loc[predicted_class]

    return predicted_class, predicted_log_proba


if __name__ == "__main__":
    
    # Read data
    FILENAME  = "dataset/simple_nb_dataset_01.csv"
    data = readData(filename = FILENAME)

    # Train data
    log_prior, word_likelihood_matrix, _ = trainNaiveBayes(data = data, 
                                                           alpha = 1.0)

    # Save model 
    #log_prior.to_pickle("dataset/naive_bayes/log_prior.pickle")
    #word_likelihood_matrix.to_pickle("dataset/naive_bayes/word_likelihood_matrix.pickle")

    # Load model
    #log_prior = pd.read_pickle("dataset/naive_bayes/log_prior.pickle")
    #word_likelihood_matrix = pd.read_pickle("dataset/naive_bayes/word_likelihood_matrix.pickle")

    # Test model
    TEST = ["predictable with no fun"]
    predict_class, predict_log_likelihood = predictNaiveBayes(test = TEST, 
                                                              log_prior = log_prior, 
                                                              word_likelihood_matrix = word_likelihood_matrix)
    predict_proba = np.exp(predict_log_likelihood[0])

    print("")
    print(f"Test: {TEST[0]}")
    print(f"Predicted class          : {predict_class[0]}")
    print(f"Predicted log likelihood : {predict_log_likelihood[0]:.4f}")
    print(f"Predicted proba          : {predict_proba:.4e}")
    print("")
    

    