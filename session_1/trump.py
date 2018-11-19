#
# trump.py
# Machine Learning Course
# Session 1  - Case Study 2
# TRUMP Chatbot
#

import pickle
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import nltk
import random
import spacy
from nltk.corpus import nps_chat
from scipy.spatial import distance

if __name__ == "__main__":
    # Load data
    debate_df = pd.read_csv("data/trump_qa.csv")
    
    statements = debate_df["Statement"].values
    responses = debate_df["Response"].values

    # Load the statement matching model we trained earlier
    with open("matcher.pickle", "rb") as f:
        model = pickle.load(f)

    # Convert statments into vectors
    nlp = spacy.load('en_core_web_sm', disable=["tagger", "ner", "textcat"])
    statement_vecs = [ nlp(statement).vector for statement in statements ]

    while True:
        user_statement = input("chat> ")
        # Exit if user requests exit
        if user_statement == "exit": break
        
        # Convert user input to vectors 
        user_statement_vec = nlp(user_statement).vector
        
        # Compute cosine distances for between the statement and the users 
        # statement
        distances = [ distance.cosine(user_statement_vec, v) for v in statement_vecs ]
        distances = np.reshape(distances, (-1, 1))

        # Compute simliarity predictions for sentence vectors
        simliarity_predicts = model.predict(distances)

        # Find the statement with the highest simliarity
        max_score = 0.0
        best_statement_idx = 0
        for i in range(len(statements)):
            score = simliarity_predicts[i]
            
            if score > max_score:
                max_score = score
                best_statement_idx = i
        # Output the response to statement with the highest simliarity
        response = responses[best_statement_idx]
        print(statements[best_statement_idx])
        print(response)
