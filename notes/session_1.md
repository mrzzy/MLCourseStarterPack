# Session I - Fundermentals of Machine Learning

## What is Machine Learning? 
**Machine Learning** (ML) is the ablity to learn without being _explictly programmed_.

For example, when you manually program a BMI calculator, you need to figure out that BMI is computed as follows:
$$
b = \frac{w}{h^2}
$$
where $w$ is the persons weight in kilograms, $h$ is the persons height in meters and $b$ is the computed BMI.


However lets say that you have never been taught how to compute BMI, only given many examples of BMI computed from weights and heights, can we still build something that will compute the BMI without knowing the formula? 

Yes, with ML we can learn the relationship/trend between BMI, weight and height and build something that computes the BMI. That is what it means to have the ability to learn without being _explictly programmed_: we do not need to know what the formula is.

## Case Study 1 - Boston Housing Prediction Problem
We will be using the boston housing price prediction problem as a running  example for the first part of this session. The problem states that given the number rooms of a house in boston, predict the price of the house.

#### Sourcing the Data
Once we have defined the problem, we need to find data to do ML on. Just like a student can't learn how to perform long division without lots of practice and examples, ML algorithms require lots of data to effectively learn from the data.

Where to find data for your ML project:
1. [Google Dataset Search](https://toolbox.google.com/datasetsearch)
2. [Kaggle](https://www.kaggle.com/)
3. [UCI ML Repository](http://archive.ics.uci.edu/ml/index.php)

For our boston housing prediction problem, Ive already done the research and have found a suitable dataset on Kaggle: [Boston Housing Dataset](https://www.kaggle.com/schirmerchad/bostonhoustingmlnd).

> If you have downloaded the MLCourse starter pack, you do not need to download data from Kaggle as the data is included in the starter pack.

### The Data Science Tookit
Time to learn the tools of the trade when doing data science in python, **numpy**, **pandas** and **matplotlib** are indespensible tools of your data science toolkit that you need to become familar to become a effcient practioner of data science.

First start your python interepter. Before we start, a small tibit: The normal python interpreter does not have colors and completion. However, if I run `ipython` instead of `python`, I get syntax coloring and when I press tab, I get completion.

#### Loading the data with pandas
Lets load **numpy**, **pandas** and **matplotlib** so that we can use them later in our code.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
  
The boston housing data that we have obtained is in the CSV format. 
> If that is giving you Vietnam War flashbacks to the time you had to read a CSV in Coursemology, please that bare with me, its much simplier with **pandas**

To load the data with pandas, all we have to do is:
```python
housing_df = pd.read_csv("data/housing.csv")
```
We have loaded the data into what pandas calls a _dataframe_.

Lets see how the data is structured by looking at the first five rows of data:
```python
housing_df.head()
```

Each row describes a house. Each column of the row describes an attribute or **feature** of the house:
| Column Name | Description |
| --- | --- |
| `RM` | number of rooms per house. |
| `LSTAT` | lower status of the population (percent). |
| `PTRATIO` | pupil-teacher ratio by town. |
| `MEDV` | price of the house . |


#### Visualising the Data with matplotlib
Unless you are a numeric demigod, its hard to make sense of long list of numbers and to see what it represents. That is what graphs are for, to visualise otherwise meaningless figures. But put away that ruler and frown because drawing graphs with **matplotlib** is as easy as it gets.

Lets say we want to observe how the price of the house (**MEDV**) varies as the number of rooms (**RM**) increases. Simply do:

1. Sort the data in the dataframe by the  order where the number of rooms increases.
```python
housing_df = housing_df.sort_values(by="RM", ascending=True)
```

2. Extract the data columns of interest from the dataframe into numpy arrays (AKA lists of numbers, with more features).
```python
rooms = housing_df.loc[:, "RM"].values
prices = housing_df.loc[:, "MEDV"].values
```

3. Plot the data on a graph with red crosses using **matplotlib**. Its also nice to add labels to graph so that other people will understand what they are looking at when you provide them the graph.
```python
plt.title("Price of House as No. Rooms increases")
plt.xlabel("No. of rooms")
plt.ylabel("Housing Price in thousands")
plt.plot(rooms, prices, "rx") 

plt.show()
```

![boston visualisation results](attachments/2b456c8e.png)

From the graph, we can sort of see a trend: as the number of rooms increases, the house price increases.


### Fundermentals of Machine Learning
Once we have our data sorted  out, we can commence on working on the ML part. From eyeballing the graph, we can sort of make out a trend, but how do get the computer to learn the trend? In this task, we are trying to model the trend by getting the computer to learn the a **model** from the **features** of the house.

Before we continue, I have implemented some utility functionality that we will use later that you should import now with:
```python
from ml_util import *
from boston import *
```

#### What is a good fit?
Similiar to purchasing clothes that fit you, we want to find a **model** that fits the data well. That is, we want to find a model that accurately reflect the trends in the data.

![Good and Bad Fits](attachments/17f5ea8d.png)

> The models are plotted by generating predictions from the model, then plotting the predictions. The predictions represent the model in the graph.

#### Measuring cost with a Cost Function
Although we could visually analyise how good or bad the model fits the data, computers can't plot graphs and determine the fit. Instead, we provide a **cost function**  (sometimes called error or loss function), a special taliors measuring tape composed of fancy mathematics that lets the computer accurately determine the fit of the data. The cost function we are using is called **mean sqaured error** or **mse** for short.

Lets see how the computed cost varies with the fit to the data. The cost function takes in the predictions from the model and the actual values, which in this case are the actual  house prices.

```python
good_predicts = predict(good_model, rooms)
good_cost = compute_cost(predicts, prices)

bad_predicts = predict(bad_model, rooms)
bad_cost = compute_cost(bad_predicts, prices)

print("good model cost: ", good_cost)
print("bad model cost: ", bad_cost)
```

| Model | Cost computed |
| --- | --- |
| `good_model` | 3481696625000.0 |
| `bad_model` | 385379267645000.0 |

As you can see, the better the fit to the data, the lower the cost computed. Hence the computer is able to measure fit by computing the cost with the cost function.

#### Training model to fit the data
Now that we are able to determine whether a model is good or bad, we can attempt to find the model that best fits the data. In general terms, when we **train/fit** a model to the data, we are instructing the computer to try and find the best model it can.

More concretely, we are asking the computer to try and minimise the cost function using an **optimizer** (the specifics  are outside the scope of this course). Recall that the lower the cost computed by the cost function, the better the fit to the data. Thus by minimising the cost function, the computer will find the model that best fits the data .

> For the morbidly curious, Gradient Descent with Momentum was used as the optimizer.

Tell the computer to find train/fit the model to the data:
```python
model = train(rooms, prices, callback=plot_progress)
```

While training, I sneaked in some snapshots of training process so that can have a look:
```python
plt.show()
```
![Training Progresss](attachments/5cb15346.png)

As you see, the model slowly finds its way to fit the data well as training progresses. The cost also, as expected, decreases as training progresses. _Generally_, the cost should decrease during training, or you are not training your model properly. (costs may temporary increase, but should, over the long term, decrease over time)

Lets plot the trained model to visualise it better:
```python
predicts = predict(model, rooms)
plt.title("LR model of Housing Price vs No. Rooms")
plt.xlabel("No. of rooms")
plt.ylabel("Housing Price in thousands")
plt.plot(rooms, prices, "rx", label="Data")
plt.plot(rooms, predicts, "g-", label="Model")
plt.legend()
plt.show()
```
![First Trained Model](attachments/865d30d7.png)

The algorithm that we have just used to train the model is called **Linear Regression**.

#### Measuring Model performance with Cross Validation
Once we have trained our model, the next step is the test the models perfomance, as to see whether we have built a good model or not. Well, lets take our data and feed it into our, model, make some predictions and compare it to the  actual answers. Why is this a bad idea?

The same way its a bad idea for a teacher to reuse homework questions in a test, we should not use the data we have used to train the model to also test the model. Why? It comes down to what we really want measure when we measure performance: we don't just want the model to succeed on the data we have, but also new, real-world examples that the model has never seen before. We want the model to **generalise** to the underlying phenomina.


To achieve this, we have to test the model on data it has never seen before, but already trained the model on all the data.  Well, that model is not we want so we discard it. Instead, lets retrain the model from scratch.

This time, we split the data into 2 sets, the **training set** & **test set**, shuffling before spliting the dataset:
```python
rooms, prices = shuffle(rooms, prices)
train_rooms, train_prices, test_rooms, test_prices = split_test_train(rooms, 
                                                                      prices, 
                                                                      0.7)
```

> The $0.7$ parameter in the `split_test_train` function tells it to split 70% of the data to training set, 30% to the test set.

> **NOTE**: please shuffle your data before you split your data to avoid some nasty nightmares. Its also important to shuffle the inputs and outputs in the **same order**.
> This is what happens if I forget to shuffle and train the model on unshuffled data:
> ![That moment when you forgot to shuffle...](attachments/332cd151.png)
> The model does fit the data because its training data is biased. 
> To remove the bias, **shuffle**.

Now we can train the model on the training set:
```python 
model = train(train_rooms, train_prices, callback=display_progress)
```

Lets find out how well our model did by testing it with the test and measuring how much it was off on average:
```python
n_test = len(test_prices)
predicts = predict(model, test_rooms)
diffs = abs(predicts - test_prices)
mean_diff = sum(diffs) / n_test
print("On average, the model is off by +- ${:.2f}".format(mean_diff))
```
`On average, the model is off by +- $84777.65`

> Wait, you can subtract two lists (`predicts - test_outs`) together?
> If they are numpy arrays and they are of the same dimensions (length), you can. What `predicts - test_outs` does in  is the same as the following long piece of code:
> ```python
> n_test = len(test_prices)
> mean_diff = []
> for i in range(n_test):
>     mean_diff.append(predicts[i] - test_prices[i])
> ```
> That is that it applies the operation to every cressponding element of the two lists and returns a new list with the results
> 

#### Overfitting and Underfitting
Our model does okay~. Sure it predicts the house price somewhat well but I think we can do better. Lets plot our model again and attempt to diagnose why:

![Underfitting diagnosis](attachments/eafba468.png)

If you look closely the data, it looks kind of curved. However, our **Linear Regression** model is only a straight line. The model does not have enough flexibility to to fit the data well because, well because a straight line does not have much flexibility at all. We need a model with more _expressive power_. This is a case of **underfitting**.

Time to use a more expressive model. This time we are using an algorithm called the **Support Vector Machine** or **SVM**.  Do worry, you don't have to understand how it works, as someone has already implemented it for you in **scikit-learn**. **scikit-learn** is a machine learning library that provides ready to eat, just add hot water algorithms for machine learning.

Lets that take that more powerful **SVM** for a spin and train a model:
```python
from sklearn.svm import SVR

model = SVR(C=1e+6, gamma=3e-3)
model.fit(train_rooms.reshape(-1, 1), train_prices)
```
> `.fit()` is scikit-learn's training function.

> Also note that we have to reshape the training inputs as `.fit()` expects it input as a 2D list/matrix in the shape (no. of examples, no. of features).
> In this case are converting an list of 382 elements into a 382 by 1 list/matrix. (-1 represents the prior length of the list).

> For now, pretend you didn't see `C` and `gamma` hyperparameters. 
 
Lets see whether that change has improved things by cross validating the **SVM** model on test data:
```python
n_test = len(test_prices)
predicts = model.predict(test_rooms.reshape(-1, 1))
diffs = abs(predicts - test_prices)
mean_diff = sum(diffs) / n_test
print("On average, the model is off by +- ${:.2f}".format(mean_diff))
```
`On average, the model is off by +- $104923.47`

Comparing the average inaccuracy of the two models:
| Type of model | Average inaccuracy in dollars
| --- | --- |
| Linear Regression | $84777.65 |
| SVM | $104923.47 |

Wait what? We changed into a more flexible, more powerful model. This means that it should by right fit the data better compared to the straight line model that we used prior to switching. Yet our cost has increased, why is that the case?

As usual, lets plot the SVM model to diagnose the situation:
```python
predicts = model.predict(rooms.reshape(-1, 1))

# reorder is required because we shuffled the data
test_rooms, test_prices = reorder((test_rooms, test_prices), by_col=0)
train_rooms, train_prices = reorder((train_rooms, train_prices), by_col=0)
rooms, predicts = reorder((rooms, predicts), by_col=0)

plt.title("SVM model of Housing Price vs No. Rooms")
plt.xlabel("No. of rooms")
plt.ylabel("Housing Price")
plt.plot(test_rooms, test_prices, "yx", label="Test Data")
plt.plot(train_rooms, train_prices, "rx", label="Training Data")
plt.plot(rooms, predicts, "g-", label="Model")
plt.legend()
plt.show()
```

![SVM overfitting](attachments/cb8a3b6b.png)

What wrong with this model? The SVM model is way too _tryhard_ and flexible. The model is trying to zigzag its way through and fit every point. The model is no longer **generalising** to the general trend in the data, instead it is starting to memorise the position of every single training data point. No doubt the model fits the training data very well, but its missing all these test data because its fitting the training data _too well_. That not a good thing as our average inaccuracy is higher than the one for the Linear Regression model we built earlier. This is a case of **overfitting**.

We want to end up in the middle ground between _Yin and Yang_, a model that neither overfits nor underfits:
```python
model = SVR(C=1e+6, gamma=1e-1)
model.fit(train_rooms.reshape(-1, 1), train_prices)
```

Cross validate to measure our performance:
```python
n_test = len(test_prices)
predicts = model.predict(test_rooms.reshape(-1, 1))
diffs = abs(predicts - test_prices)
mean_diff = sum(diffs) / n_test
print("On average, the model is off by +- ${:.2f}".format(mean_diff))
```
`On average, the model is off by +- $79114.70`

Thats better, the SVM model now does better than the Linear Regression Model:
| Type of model | Average inaccuracy in dollars
| --- | --- |
| Linear Regression | $84777.65 |
| SVM | $79114.70 |

Plotting the SVM model, we can clearly visualise why the average inaccuracy has improved:
```python 
plt.clf()
predicts = model.predict(rooms.reshape(-1, 1))

rooms, predicts, prices = reorder((rooms, predicts, prices), by_col=0)
plt.title("SVM model of Housing Price vs No. Rooms")
plt.xlabel("No. of rooms")
plt.ylabel("Housing Price")
plt.plot(rooms, prices, "rx", label="Data")
plt.plot(rooms, predicts, "g-", label="Model")
plt.legend()
plt.show()
```

![SVM good fit](attachments/fafff174.png)
Ah we have found the balance: this is the fitting middle ground we want. Not too simple that we aren't able to capture the trends in the data, yet not too complex that it starts to memorise the training data.

5 min break.
  
## Case Study 2 - Chatbot
Time to put theory into practice by building a Chatbot. We should be able type a statement and have it reply to it. 

### Applying Machine Learning to Projects
For every project, we must first do some planning on how we are going about to implement it:

A rough sketch of the design is given below: 
1. obtain a statement from the user
2. match the given statement against our list of statements
3. output the Trump's response to that statement

From the design  sketch, stages 1. and 3. seem trival to implement. But stage 2. seems quite difficult.

How shall we match 2 statments with similar meanings together, although they might not contain the same words? For example:
_"Three friends are making faces for the camera"_
_"Three friends are making faces for a photo"_
Are two different ways of saying the same thing.

One possible solution is to apply machine learning to build a model that would rate the similarity score for two sentences, which we will then use to compare the user statement with our list of statements. The statement match would happen with the statement with the highest similarity score. Finally, we just return the reponse to the best matching statement to the user.

#### Sourcing and loading the Data
As with the previous case study, now would be the time to do some research online for a suitable dataset. For this example, since we are trying to build a model that produces simliarity scores for two sentences, our criteria would be to find a dataset that provides sentence pairs and similarity scores. 

In the interest of time and brievity, I have included the dataset in the starter pack. Lets load the data with pandas and take a look:
```python
similarity_df = pd.load_csv("data/similarity.tsv", sep="\t")
```
> Instead of being a CSV file, the data is formmated as a TSV (Tab seperated values  file).
> Thus, we use the `sep` parameter to tell the function that we are reading a TSV file.

Taking a look at the first few rows of the data:
```python
similarity_df.head()
```

| Column | What does it represent |
| --- | --- |
| pair_ID | Identifier no. of the sentence pair |
| sentence_A | The first sentence of the pair |
| sentence_B | The second sentence of the pair |
| related_score | On a scale of 0.0 to 5.0, how related are the two sentences? |
| entailment_judgement | Whether the two sentences contradict each other |

#### Preprocessing the Data
Frequently the raw data from the dataset is not in a suitable format to conduct machine learning with. For example, you cannot conduct machine learning directly on text data. We have to convert the text to a series of numbers before we conduct machine learning. We have to apply **data preprocessing** to the data :

We dont want to train on the sentences that contradict each other so lets remove them:
```python
non_contradict_indexes = similarity_df["entailment_judgment" != "CONTRADICTION"]
similarity_df = simliarity_df[non_contradict_indexes]
```


Extract the data columns we are interested in:
```python
n_data = 10000
sentence_As = similarity_df.loc[:, "sentence_A"].values[:n_data]
sentence_Bs = similarity_df.loc[:, "sentence_B"].values[:n_data]
similarity_scores = similarity_df.loc[:, "relatedness_score"].values[:n_data]
```

##### Textual Feature extraction 
As mentioned earlier, cannot conduct ML on text directly. We must convert the text to a series of numbers representing the text. This process is called **feature extraction**: representing the data in another form (ie text to numbers). Thankfully, spacy provides a way to convert text to vectors:

Using **spacy**,, convert the text into a series of numbers (vectors) so that we can perform machine learning on them:
```python
import spacy

nlp = spacy.load('en_core_web_sm', disable=["tagger", "ner", "textcat"])
vector_As = [ nlp(sentence).vector for sentence in sentence_As ]
vector_Bs = [ nlp(sentence).vector for sentence in sentence_Bs ]
```

#### Feature engineering with Cosine distance
Since we are comparing 2 sentences, we compute the cosine distance between them:
```python
from scipy.spatial import distance

distances = [ distance.cosine(vec_a, vec_b) \
             for vec_a, vec_b in zip(vector_As, vector_Bs) ]
distances = np.reshape(distances, (-1, 1))
```

##### Test Train split
As usual, split the dataset into training and test sets for later cross validation.
This time however, we will be using **scikit-learn**'s function to do the splitting:
```python
train_sents, test_sents, train_scores, test_scores = \
    train_test_split(sentence_pairs, similarity_scores, test_size=0.2)
```
> `train_test_split()` automatically shuffles for us before splitting.

#### Training a Model II: Hyperparameter Tuning

Before we train the model, remember the `C` and `gamma` parameters that we ignored earlier?
```python
model = SVR(C=..., gamma=...)
```
These are called **hyperparameters**. Unfortunately, there are no quick and easy way to find values for your hyperparameters. We will use the tried and tested, good old guess and check: pick random value and check if your model improves by computing the cost using the cost function.

Picking a combination of random hyperparameters `C` and `gamma` we will train a model. We will then cross validate by computing the cost, keeping the model with the best cost:
```python
import sys
from sklearn.metrics import mean_squared_error
from random import random, randint

best_model = None
best_cost = 9e+9

for i in range(60):
    # Randomly select hyperparameters
    c_val = random() * 10 ** randint(-3, 6)
    gamma_val = random() * 10 ** randint(-5, 1)

    # Train model with random hyperparameter
    model = SVR(C=c_val, gamma=gamma_val)
    model.fit(train_vecs, train_scores)

    # Cross validate with test set by computing cost
    predict_scores = model.predict(test_vecs)
    cost = mean_squared_error(test_scores, predict_scores)

    # Display a dot every model trained
    print(".", end="")
    sys.stdout.flush()

    # Save model if best model so far
    if cost < best_cost:
        print("Found better model: C={:.2e}, gamma={:.2e}".format(c_val, gamma_val))
        print("cost: ", cost)
        best_model = model
        best_cost = cost
```
> **NOTE**: In the interest of time, please do not run this code to search for good hyperparameters
> Replace with the following code instead:
> ```python
> best_model = SVR(C=7.59e+02, gamma=4.82e-02)
> ```


##### Saving your model for later use
Now that we have model we want, lets save our work so that we can use to to build our statement matching component of our Chatbot:

```python
import pickle

pipeline = {
    "vectorizer": vectorizer,
    "model": best_model
}

with open("matcher.pickle", "wb") as f:
    pickle.dump(pipeline, f)
```

### Trying out your chatbot
```sh
python trump.py
```