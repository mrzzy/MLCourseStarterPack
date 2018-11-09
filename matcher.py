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

    # Shape the data into the format we want:
    # sentence pairs and simlairity scores
    sentence_As = similarity_df.loc[:, "sentence_A"].values
    sentence_Bs = similarity_df.loc[:, "sentence_B"].values
    sentence_pairs = sentence_As + " " +  sentence_Bs

    similarity_scores = similarity_df.loc[:, "relatedness_score"].values

    # Split the data into test and train sets
    train_sents, test_sents, train_scores, test_scores = \
        train_test_split(sentence_pairs, similarity_scores, test_size=0.2)

    # Vectorize the sentences into vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(train_sents) 
    train_vecs = vectorizer.transform(train_sents)
    test_vecs = vectorizer.transform(test_sents)
    
    # Train the models with random hyperparameters, saving the best model
    #best_model = None
    #best_cost = 9e+9
    #
    #for i in range(60):
    #    # Randomly select hyperparameters
    #    c_val = random() * 10 ** randint(-3, 6)
    #    gamma_val = random() * 10 ** randint(-5, 1)

    #    # Train model with random hyperparameter
    #    model = SVR(C=c_val, gamma=gamma_val)
    #    model.fit(train_vecs, train_scores)
    #    
    #    # Cross validate with test set by computing cost
    #    predict_scores = model.predict(test_vecs)
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
    best_model = SVR(C=7.59e+02, gamma=4.82e-02)
    best_model.fit(train_vecs, train_scores)
    predict_scores = best_model.predict(test_vecs)
    cost = mean_squared_error(test_scores, predict_scores)
    print(cost)

    # Save trained matcher pipeline: vectorizer, model for use later
    pipeline = {
        "vectorizer": vectorizer,
        "model": best_model
    }
    
    with open("matcher.pickle", "wb") as f:
        pickle.dump(pipeline, f)
