#
# boston.py
# Machine Learning Course
# Session 1
# Boston Housing Price Prediction
#

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import time

## Utilities
# Append a column of ones to the inputs
# Returns the updated inputs
def append_ones(inputs):
    inputs = np.c_[inputs, np.ones((len(inputs),))]
    return inputs


# Evalute the predictions given model with the given inputs
# Returns the predictions computed fro the given model
def predict(model, inputs):
    # Add ones to inputs to treat constant term of model as just another feature
    inputs = append_ones(inputs)
    # Compute predictions
    return np.dot(model, inputs.T)

# Computes the mean square error for the given predictions predicts and actual
# values acutals
# Returns the computed mean squared error
def compute_cost(predicts, actuals):
    return sum((predicts - actuals) ** 2) / 2

# Computes the gradients for mean sqaured error for the given inputs, predictions
# predicts and actual values actuals
# Returns the computed gradients
def compute_gradients(inputs, predicts, actuals):
    n_data = len(inputs)
    
    # Add ones to inputs to treat constant term of model as just another feature
    inputs = append_ones(inputs)
    # Compute gradients
    gradients = np.dot((predicts - actuals),  inputs) / n_data

    return gradients

# Generate polynomial data for a given single scalar inputs into an input matrix 
# of n degrees, each as a single feature.
# Returns the converted input matrix
def polynomialize_data(inputs, n_degrees):
    input_matrix = np.zeros((len(inputs), n_degrees))
    for i in range(n_degrees):
        input_matrix[:, i] = inputs ** (n_degrees - i)
    
    return input_matrix

## Training
# Progress Callback to pretty dipslay the current training progress to stdout
def display_progress(n_iter, i_iter, model, inputs, actuals):
    # Output cost over training iterations
    predicts = predict(model, inputs)
    cost = compute_cost(predicts, actuals)
    print("{}/{} - cost: {:.3f}".format(i_iter + 1, n_iter, cost))

# Progress Callback to show a graph to display the current training progress
# every given n_step iterations
i_plot = 0
def plot_progress(n_iter, i_iter, model, inputs, actuals, n_step=125):
    global i_plot
    if i_iter % n_step == 0:
        i_plot = i_plot % (n_iter // n_step) + 1
        plt.subplot(2, 4, i_plot)

        # Plot the current model with the data to visualise performance
        predicts = predict(model, inputs)
        plt.title("{}. Training Model - {}/{}".format(i_plot, i_iter, n_iter))
        plt.xlabel("No. of rooms")
        plt.ylabel("Housing Price in thousands")
        plt.plot(inputs, actuals, "r+", label="Data")
        plt.plot(inputs, predicts, "g-", label="Model")
        plt.legend()
        
        # Display current cost for model
        predicts = predict(model, inputs)
        cost = compute_cost(predicts, actuals)
        plt.suptitle("Cost: {:.2f}".format(cost)) 
        

# Train a multiple linear regression model on the given data as inputs and 
# actuals using gradient descent with momentum with the given learning_rate for 
# the given number of iterations n_iter with the given momentum rate.
# If provided will call the given progress callback every iteration of training 
# provides the callback no. of iteration, current iteration and current model
# and the data as inputs and ouptus
def train(inputs, actuals, learning_rate=0.3, momentum_rate=0.95, 
          n_iter=10000, callback=None):
    # Train model with parameter for every feature in input
    n_feature = inputs.shape[1] if len(inputs.shape) > 1 else 1
    model = [ 0.0 ] * (n_feature + 1)

    # Minimise cost with gradient descent
    prev_step = np.zeros(1)
    for i_iter in range(n_iter):
        # Compute gradient descent step
        predicts = predict(model, inputs)
        gradients = compute_gradients(inputs, predicts, actuals)
        
        # Augment gradient descent step with momentum from previous step
        if prev_step.any():
            step = (momentum_rate * prev_step)  + \
                    ((1.0 - momentum_rate) * learning_rate * gradients)
        else:
            step = learning_rate * gradients
        
        # Perform gradient descent step
        model -= step
        prev_step = step
        
        # Call progress callback
        if callback: callback(n_iter, i_iter, model, inputs, actuals)
    
    return model


# Good and bad models to teach the concept of fit
good_model = np.asarray([2e+5, -8e+5])
bad_model = np.asarray([4e+5, -8e+5])

# TODO: Development test code, remove on course deployment
if __name__ == "__main__":
    # Load housing data
    housing_df = pd.read_csv("data/housing.csv")
    
    # Visualise data with Scatterplot of ascending no. room against the price
    housing_df = housing_df.sort_values(by="RM", ascending=True)
    rooms = housing_df.loc[:, "RM"].values
    prices = housing_df.loc[:, "MEDV"].values
    plt.title("Price of House as No. Rooms increases")
    plt.xlabel("No. of rooms")
    plt.ylabel("Housing Price in thousands")
    plt.plot(rooms, prices, "rx") 

    #plt.show()

    # Show that the cost function computes the fit to the data
    # Good model
    good_predicts = predict(good_model, rooms)
    good_cost = compute_cost(good_predicts, prices)
    print("good cost:", good_cost)
    # Bad model
    bad_predicts = predict(bad_model, rooms)
    bad_cost = compute_cost(bad_predicts, prices)
    print("bad cost:", bad_cost)

    # Training the model
    model = train(rooms, prices, learning_rate=0.3, momentum_rate=0.98, 
                  n_iter=10 ** 3, callback=plot_progress)
    
    plt.show()
    # Visualise trained model
    predicts = predict(model, rooms)
    plt.title("LR model of Housing Price vs No. Rooms")
    plt.xlabel("No. of rooms")
    plt.ylabel("Housing Price in thousands")
    plt.plot(rooms, prices, "rx", label="Data")
    plt.plot(rooms, predicts, "g-", label="Model")
    plt.legend()
    plt.show()

