#
# ml_util.py
# Machine Learning Course
# Machine Learning Utilities
#

import random
import math
import numpy as np

# Shuffle datasets inputs and outputs.
# Inputs and outputs would be shuffled in the same order, so any ordering relation 
# between coresponding inputs and outputs will be preserved.
# input and output length must be the same.
# Returns the shuffled inputs and outputs
def shuffle(inputs, outputs):
    if not len(inputs) == len(outputs): 
        raise ValueError("Input and outs length are different")

    n_data = len(inputs)

    # Conduct shuffle on copy
    shuffle_ins = inputs[:]
    shuffle_outs = outputs[:]

    # Fisher-Yates shuffle
    for i in range(n_data - 1): # 1 to n - 2
        j = random.randint(i, n_data - 1) # random j that satisfies i <= j <= (n - 1)
        # Swap jth element with ith element to place jth element in randomized
        # subsection of array
        shuffle_ins[i], shuffle_ins[j] = shuffle_ins[j], shuffle_ins[i] 
        shuffle_outs[i], shuffle_outs[j] = shuffle_outs[j], shuffle_outs[i] 
    
    return shuffle_ins, shuffle_outs

# Reorder the given data columns by the specifed column by_col
# Returns the sorteed data columns
def reorder(data_cols, by_col=0):
    n_col =  len(data_cols)
    # Combine and sort data columns
    dataset = zip(*data_cols)
    dataset = sorted(dataset, key=(lambda e: e[by_col]))
    
    # Extract out columns and revert to numpy arrays
    data_cols = zip(*dataset)
    data_cols = [ np.asarray(c) for c in data_cols ]
    
    return data_cols

# Test Train Spliter
# Splits input and output data into test and train data in ratio 7:3
# input and output length must be the same.
# Returns (train_ins, train_outs, test_ins, tests_outs)
def split_test_train(inputs, outputs, ratio=0.7):
    if not len(inputs) == len(outputs): 
        raise ValueError("Input and outs length are different")
    
    border = math.floor(ratio * len(inputs))
    # Demarking a 80% slice of data for training
    train_ins = inputs[:border] # slice from start to (border - 1)
    train_outs = outputs[:border] # slice from start to (border - 1)
    
    test_ins = inputs[border:] # slice from border to end
    test_outs = outputs[border:] # slice from border to end

    return (train_ins, train_outs, test_ins, test_outs)

