# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <h1> This project discusses beginner level text classification using a Alexa product reviews dataset off Kaggle. </h1>
# <h2> The dataset contains 3000 reviews of Alexa products. </h2>
# <p> The notebook deals with basic text classification tasks including the preprocessing of text data, vectorization of text data, and the application of machine learning algorithms to classify the reviews into positive and negative categories using Logistical Regression which is the stepping stone towards more deeper and intricate models. </p>
# <p> We will also use NLP libraries like NLTK to preprocess the text data and vectorize it. </p>

# <h3> Step 1: Tech Stack setup and installing dependencies </h3>
# <p> The first step is to download the nltk library and install the dependencies required for the project. The nltk library is a powerful library for natural language processing and text classification tasks. A short description of the libraries is given below: </p>
# <ul>
#     <li> nltk: The Natural Language Toolkit is a powerful library for text processing and classification tasks. </li>
#         <ul>
#             <li> punkt: The punkt module is used for tokenization of text data. </li>
#             <li> stopwords: The stopwords module is used for removing stopwords from text data. </li>
#             <li> wordnet: The wordnet module is used for lemmatization of text data. </li>
#             <li> omw-1.4: The omw-1.4 module is used for wordnet synsets. </li>
#         </ul>
#     <li> numpy: The numpy library is a powerful library for numerical computing and array manipulation. </li>
#     <li> pandas: The pandas library is a powerful library for data manipulation and analysis. </li>
#     <li> scikit-learn: The scikit-learn library is a powerful library for machine learning and data analysis. </li>
#     <li> matplotlib: The matplotlib library is a powerful library for data visualization and plotting. </li>
# </ul>

# install dependencies if they do not exist using the requirements.txt file and pip
# %pip install -r requirements.txt

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Importing the required libraries for the project

import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# <h3> Step 3: Loading the Dataset and Exploratory Analysis </h3>
# <p> The dataset can be downloaded from Kaggle, but for convinience it is also included in the repository as a tsv file.
# Link to dataset - https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews/data</p>

# The cell below will load the dataset and help us check the first few rows.

# load the dataset as a pandas DataFrame 
reviews = pd.read_csv('amazon_alexa.tsv', delimiter='\t')
# display the first few rows of the dataset
reviews.head()

# Using the matplotlib library to plot the distribution of the ratings in the dataset for exploratory analysis.

plot = reviews['rating'].value_counts().plot(kind='bar', title='Distribution of Ratings in the Dataset')

# Displaying distribution of target variable 'feedback' in the dataset. The feedback column contains the sentiment of the review. As you can notice the data is unbalanced. We will split the data into training and test split later on while keeping this in mind.

feedback_plot = reviews['feedback'].value_counts().plot(kind='bar', title='Distribution of Feedback in the Dataset')

# <h3> Step 4: Preprocessing the Text Data </h3>
# <p> The text data needs to be preprocessed before it can be used for classification tasks. The preprocessing steps include tokenization, removing stopwords, and lemmatization. </p>
# <p> The following steps are performed: </p>
# <ol>
#     <li> Data Cleaning: Removing NaN values and duplicates from the dataset. </li>
#     <li> Tokenization: The text data is tokenized into words. </li>
#     <li> Removing Stopwords: The stopwords are removed from the text data. </li>
#     <li> Lemmatization: The words are lemmatized to their root form. </li>

# Remove NaN values and duplicates from the dataset
reviews.dropna(inplace=True)
reviews.drop_duplicates(inplace=True)

# <h3> Step 5: Splitting the Dataset into Training and Test Sets </h3>
# <p> The dataset is split into training and test sets for training and evaluating the machine learning model. </p>

# +
from sklearn.model_selection import train_test_split

# split the dataset into training and test sets with equal distribution of the target variable
X_train, X_test, y_train, y_test = train_test_split(reviews['verified_reviews'], reviews['feedback'], test_size=0.2, random_state=0, stratify=reviews['feedback'])

# display the shape of the training and test sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# -

# <h3> Step 7: Building the pipeline and training the Machine Learning Model </h3>
# <p> The machine learning model is built using the Logistical Regression algorithm. The model is trained on the training data and evaluated on the test data. The pipeline allows us to vectorize the text data and train the model in a single step. </p>

# +
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.tokenize import word_tokenize
# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Tokenization, lemmatization, and removing stopwords/punctuations function
def nltk_tokenizer(sentence):
    tokens = word_tokenize(sentence)  # Tokenize the sentence
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token not in string.punctuation 
              and token not in stopwords.words('english')]  # Lemmatize tokens and remove stopwords/punctuations
    return tokens

def clean_text(text):
    if isinstance(text, float) and pd.isnull(text):
        return ""
    return str(text).strip().lower()

X_train = X_train.apply(clean_text)
X_test = X_test.apply(clean_text)

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=nltk_tokenizer)),
    ('classifier', LogisticRegression())
])

# train the model
pipeline.fit(X_train, y_train)

# make predictions on the test data
y_pred = pipeline.predict(X_test)
# -

# <h3> Step 8: Evaluating the Model </h3>
# <p> The model is evaluated using accuracy and classification report. The accuracy score is calculated to measure the performance of the model. The classification report provides the precision, recall, and F1-score for each class in the target variable. </p>
#

# +
from sklearn.metrics import accuracy_score, classification_report

# calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# display the accuracy of the model
print(f'Accuracy: {accuracy}')

# display the classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))
# -

# <h3> Step 9 (optional): Saving the Model </h3>
# <p> The model can be saved using the joblib library for future use. The model can be loaded and used for making predictions on new data. </p>

# +
import joblib

# save the model
joblib.dump(pipeline, 'alexa_reviews_model.joblib')
# -

# <h3> Step 10 (optional): Loading the Model and giving it a test run using custom reviews </h3>
# <p> The model can be loaded using the joblib library and used for making predictions on new data. The model is tested on custom reviews to check its performance. </p>
#

# +
import joblib

# load the model
model = joblib.load('alexa_reviews_model.joblib')

# test the model on custom reviews
custom_reviews = [
    'The product is excellent and works perfectly as described.',
    'The product is not good and does not work as expected.',
    'The product is okay but could be better in terms of quality.',
    'The product is not that good. Would not recommend it to anyone.',
    'Very bad worse product ever. Would not recommend it to anyone.'
]

# make predictions on custom reviews
predictions = model.predict(custom_reviews)

# display the predictions
for review, prediction in zip(custom_reviews, predictions):
    print(f'Review: {review}')
    print(f'Prediction: {"Positive" if prediction == 1 else "Negative"}')
    print()
