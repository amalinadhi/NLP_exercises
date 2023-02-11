import numpy as np
import pandas as pd
import re
from sklearn import manifold

import matplotlib.pyplot as plt
import seaborn as sns


def readData(filename):
    data = pd.read_csv(filename,
                       header = None,
                       sep = ";")
    text = data[0].values[0]

    return text

def tokenize(text):
    # lower the text
    text = text.lower()

    # tokenize
    pattern = re.compile(r'[A-Za-z]+')
    token = pattern.findall(text)

    return token

def mapping(token):
    word_to_id = {}
    id_to_word = {}

    for id, token in enumerate(set(token)):
        word_to_id[token] = id
        id_to_word[id] = token

    return word_to_id, id_to_word

def oneHotEncode(word_id, word_size):
    ohe = [0] * word_size
    ohe[word_id] = 1

    return ohe

def generateTraining(window, token, word_to_id, random_state = 42):
    # Set random state
    np.random.seed(random_state)

    # X --> One-Hot encoding of word
    # y --> One-Hot encoding of context
    X = []
    y = []
    n_token = len(token)

    for i in range(n_token):
        # Generate current word & its context range
        current_word_idx = i

        if current_word_idx < window:
            start_range = 0
            end_range = current_word_idx + window + 1
        elif n_token-current_word_idx <= window:
            start_range = current_word_idx - window
            end_range = n_token
        else:
            start_range = current_word_idx - window
            end_range = current_word_idx + window + 1

        context_range_idx = range(start_range, end_range)
        
        #print(f"{i} - {context_range_idx}")
        
        # Generate the word
        for j in context_range_idx:
            current_context_idx = j

            # Do not do anything if word = context
            if current_context_idx == current_word_idx:
                continue
            
            # Get its current word & context
            current_word = token[current_word_idx]
            current_context = token[current_context_idx]
            #print(current_word, current_context)

            # Get the word & context id from corpus
            word_id = word_to_id[current_word]
            context_id = word_to_id[current_context]

            # Get its One-Hot Encoding
            ohe_word = oneHotEncode(word_id = word_id,
                                    word_size = len(word_to_id))
            ohe_context = oneHotEncode(word_id = context_id,
                                       word_size = len(word_to_id))
            
            # Add to X & y
            X.append(ohe_word)
            y.append(ohe_context)

    # Turn to np array
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y

def activationFunc(a2, type):
    """
    Convert a vector of K real numbers into a probability distribution of K 
    possible outcomes
    """
    if type == "softmax":
        Z = []
        for row in a2:
            exp_row = np.exp(row)
            z = exp_row / np.sum(exp_row)

            Z.append(z)

        Z = np.array(Z)

    return Z

def lossFunc(y_hat, y_act, type):
    if type == "cross_entropy":
        loss = -np.sum(y_act * np.log(y_hat))

    return loss
        
def initNetwork(n_embedding, word_size, random_state = 42):
    # Set random state
    np.random.seed(random_state)

    # Define model
    model = {
        "w1": np.random.randn(word_size, n_embedding),
        "w2": np.random.randn(n_embedding, word_size)
    }

    return model

def forwardPropagation(model, X, return_cache = True):
    # Getting value for the backpropagation
    cache = {}

    # Calculate hidden layer
    cache["a1"] = X @ model["w1"]

    # Calculate output layer
    cache["a2"] = cache["a1"] @ model["w2"]

    # Calculate activation of the output layer
    cache["z"] = activationFunc(a2 = cache["a2"],
                                type = "softmax")

    if not return_cache:
        return cache["z"]

    return cache

def backwardPropagation(model, X, y, alpha):
    # 1. Get the prediction
    y_cache = forwardPropagation(model = model,
                                 X = X)
    y_hat = y_cache["z"]

    # 2. find the gradient for gradient descent
    # need: dL/dw1 & dL/dw2
    # calculate the local gradient
    dL_da2 = y_hat - y          
    da2_da1 = model["w2"]       
    da2_dw2 = y_cache["a1"]     
    da1_dw1 = X                 

    # calculate the loss gradient with chain rule 
    # initially
    #dL_dw1 = dL_da2.T @ (da2_da1 @ da1_dw1.T).T
    #dL_dw2 = (dL_da2.T @ da2_dw2).T

    # simplified with dot product matrix properties
    # (x @ y).T = y.T @ x.T
    dL_dw1 = dL_da2.T @ da1_dw1 @ da2_da1.T
    dL_dw2 = da2_dw2.T @ dL_da2
             
    # 3. Defend
    assert(dL_dw1.shape == model["w1"].shape)
    assert(dL_dw2.shape == model["w2"].shape)

    # 4. Update with gradient descent
    model["w1"] -= alpha * dL_dw1
    model["w2"] -= alpha * dL_dw2

    # 5. Calculate loss
    loss = lossFunc(y_hat = y_hat,
                    y_act = y,
                    type = "cross_entropy")

    return loss

def fit(X, y, model_param, print_hist = False):
    # Extract model param
    n_embedding = model_param["n_embedding"]
    word_size = model_param["word_size"]
    alpha = model_param["learning_rate"]
    n_iter = model_param["n_iter"]

    # 1. Initialize the network
    model = initNetwork(n_embedding = n_embedding,
                        word_size = word_size)

    # 2. Iterate model
    loss_history = []
    for i in range(n_iter):
        # Update the model parameter
        loss = backwardPropagation(model = model,
                                   X = X,
                                   y = y,
                                   alpha = alpha)
        
        # Print loss
        if i%10 == 0 & print_hist == True:
            print(f"iter-{i+1} - loss: {loss:.4f}")

        # Append loss to lost_history
        loss_history.append(loss)

    if print_hist:
        print(f"iter-{i+1} - loss: {loss:.4f}")

    return model, loss_history

def plotLossHistory(loss_history):
    n = len(loss_history)
    x = np.arange(1, n+1)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    ax.plot(x, loss_history)
    ax.set_xlabel("iteration-i")
    ax.set_ylabel("Crossentropy loss")

    plt.show()

def getEmbedding(word, model, word_to_id):
    try:
        idx = word_to_id[word]
    except KeyError:
        print("'word' not in corpus")

    word_ohe = oneHotEncode(word_id = idx,
                            word_size = len(word_to_id))
    
    word_embedding = forwardPropagation(model = model,
                                        X = word_ohe)["a1"]
    
    return word_embedding

def calculateSimilarity(word_1, word_2, model, word_to_id):
    # Get word embedding
    word_embedding_1 = getEmbedding(word = word_1,
                                    model = model,
                                    word_to_id = word_to_id)
    word_embedding_2 = getEmbedding(word = word_2,
                                    model = model,
                                    word_to_id = word_to_id)

    # Calculate similarity using dot product
    similarity = np.dot(word_embedding_1, word_embedding_2)

    return similarity

def mostSimilar(word, kind, model, word_to_id):
    # Iterate & check similarity to all word
    similarity_vector = np.zeros(len(word_to_id))
    for i, other_word in enumerate(word_to_id):
        # Calculate similarity between word & other word
        other_word_similarity = calculateSimilarity(word_1 = word,
                                                    word_2 = other_word,
                                                    model = model,
                                                    word_to_id = word_to_id)
        similarity_vector[i] = other_word_similarity

    # Print the output
    n_word = 10
    if kind == "most_similar":
        sorted_idx = np.argsort(similarity_vector)[1:n_word+1][::-1]
    elif kind == "most_opposite":
        sorted_idx = np.argsort(similarity_vector)[1:n_word+1]
    elif kind == "most_unrelated":
        sorted_idx = np.argsort(np.abs(similarity_vector))[1:n_word+1]
    
    print(f"Top {n_word} words that {kind} to '{word}':")
    for i, idx in enumerate(sorted_idx):
        print(f" {i+1}. {list(set(word_to_id))[idx]}")

def plotEmbedding(model, word_to_id):
    # Create list of word embedding
    word_embedding_list = []
    for word in word_to_id:
        word_embedding = getEmbedding(word = word,
                                      model = model,
                                      word_to_id = word_to_id)
        word_embedding_list.append(word_embedding)

    word_embedding_list = np.array(word_embedding_list)
    
    # Transform to 2D
    tsne = manifold.TSNE(n_components = 2,
                         random_state = 42)
    Z = tsne.fit_transform(word_embedding_list)

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    for i in range(len(Z)):
        x_coor = Z[i, 0]
        y_coor = Z[i, 1]

        ax.scatter(x_coor, y_coor)
        ax.annotate(list(set(word_to_id))[i],
                    xy = (x_coor, y_coor),
                    xytext = (5, 2),
                    textcoords = "offset points",
                    ha = "right",
                    va = "bottom")
    
    plt.show()



if __name__ == "__main__":
    # 1. Read data
    FILENAME = "dataset/sample_text.csv"
    text = readData(filename = FILENAME)
    #print(text)

    # 2. Tokenization
    token = tokenize(text = text)
    #print(token)
    #print(len(token))

    # 3. Map token
    word_to_id, id_to_word = mapping(token = token)
    #print(word_to_id)
    #print(len(word_to_id))

    # 4. Generate training data
    X, y = generateTraining(window = 3,
                            token = token,
                            word_to_id = word_to_id)
    #print(X[:5])

    # 5. Train model using Neural Network
    model_param = {"n_embedding": 10,
                   "word_size": len(word_to_id),
                   "learning_rate": 0.01,
                   "n_iter": 100}
    model, loss_history = fit(X = X,
                              y = y,
                              model_param = model_param)
    
    plotLossHistory(loss_history = loss_history)

    # 6. Find most similar word
    mostSimilar(word = "algorithms",
                kind = "most_similar",
                model = model,
                word_to_id = word_to_id)
    mostSimilar(word = "algorithms",
                kind = "most_opposite",
                model = model,
                word_to_id = word_to_id)
    mostSimilar(word = "algorithms",
                kind = "most_unrelated",
                model = model,
                word_to_id = word_to_id)

    # 7. Plot word embedding
    plotEmbedding(model = model,
                  word_to_id = word_to_id)