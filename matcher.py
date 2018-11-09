#
# matcher.py
# Machine Learning Course
# Session 1  - Case Study 2
# Train Sementic Text matcher
#

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import sys
import spacy

from scipy.spatial import distance
from random import random, randint
from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    # Load the simliarity data - data is tab seperated
    similarity_df =  pd.read_csv("data/similarity.tsv", sep="\t")
    # Remove contradicting sentences
    non_contradict_indexes = similarity_df["entailment_judgment"] != "CONTRADICTION"
    similarity_df = similarity_df[non_contradict_indexes]

    # Extract the sentences
    # sentence pairs and simlairity scores
    n_data = 10000
    sentence_As = similarity_df.loc[:, "sentence_A"].values[:n_data]
    sentence_Bs = similarity_df.loc[:, "sentence_B"].values[:n_data]
    similarity_scores = similarity_df.loc[:, "relatedness_score"].values[:n_data]
    
    # Feature extraction: Vectorize the sentences into vectors
    print("Vectoring senteces...")
    nlp = spacy.load('en_core_web_sm', disable=["tagger", "ner", "textcat"])
    vector_As = [ nlp(sentence).vector for sentence in sentence_As ]
    vector_Bs = [ nlp(sentence).vector for sentence in sentence_Bs ]

    # Feature engineering: derieve cosine distances from sentences A and B
    # to make the task easier for the regressor
    print("Computing distance...")
    distances = [ distance.cosine(vec_a, vec_b) \
                 for vec_a, vec_b in zip(vector_As, vector_Bs) ]
    distances = np.reshape(distances, (-1, 1))

    # Split the data into test and train sets
    train_dists, test_dists, train_scores, test_scores = \
        train_test_split(distances, similarity_scores, test_size=0.2)

    # Train the models with random hyperparameters, saving the best model
    print("selecting hyperparamters...")
    best_model = None
    best_cost = 9e+9
    
    #def search_hyperparameters(i):
    #    global best_model
    #    global best_cost

    #    # Randomly select hyperparameters
    #    c_val = random() * 10 ** randint(-3, 6)
    #    gamma_val = random() * 10 ** randint(-5, 1)

    #    # Train model with random hyperparameter
    #    model = SVR(C=c_val, gamma=gamma_val)
    #    model.fit(train_dists, train_scores)
    #    
    #    # Cross validate with test set by computing cost
    #    predict_scores = model.predict(test_dists)
    #    cost = mean_squared_error(test_scores, predict_scores)
    #    
    #    # Display a dot every model trained
    #    sys.stdout.flush()
    #    print(".", end="")

    #    # Save model if best model so far
    #    if cost < best_cost:
    #        print("Found better model: C={:.2e}, gamma={:.2e}".format(c_val, gamma_val))
    #        print("cost: ", cost)
    #        best_model = model
    #        best_cost = cost
    #
    #from multiprocessing import Pool, cpu_count
    #pool = Pool(cpu_count())
    #pool.map(search_hyperparameters, range(20))

    
    best_model = SVR(C=7.64e+05, gamma=9.44e-03)
    best_model.fit(train_dists, train_scores)
    predict_scores = best_model.predict(test_dists)
    cost = mean_squared_error(test_scores, predict_scores)

    # Save trained matcher model for use later
    with open("matcher.pickle", "wb") as f:
        pickle.dump(best_model, f)
