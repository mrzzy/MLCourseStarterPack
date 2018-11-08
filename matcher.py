#
# matcher.py
# Machine Learning Course
# Session 1  - Case Study 2
# Train Sementic Text matcher
#
import sys

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from random import random, randint
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
    
    # Feature scaling to accelerate learning
    scaler = StandardScaler(with_mean=False)
    scaler.fit(train_vecs)
    train_vecs = scaler.transform(train_vecs)
    test_vecs = scaler.transform(test_vecs)
        
    # Train the models with random hyperparameters, saving the best model
    Cs = [ random() * 10 ** randint(-2, 3) for i in range(8) ]
    gammas = [ random() * 10 ** randint(-5, 1) for i in range(6) ]

    best_model = None
    best_cost = 9e+9
    
    for c_val in Cs:
        for gamma_val in gammas:
            # Train model with random hyperparameter
            model = SVR(C=c_val, gamma=gamma_val)
            model.fit(train_vecs, train_scores)
            
            # Cross validate with test set by computing cost
            predict_scores = model.predict(test_vecs)
            cost = mean_squared_error(test_scores, predict_scores)
            sys.stdout.flush()

            # Save model if best model so far
            if cost < best_cost:
                print("Found better model: C={:.2e}, gamma={:.2e}".format(c_val, gamma_val))
                print("cost: ", cost)
                best_model = model
                best_cost = cost

