"""
Classifier Trainer
Project: Disaster Response

Sample Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

Sample Script Execution:
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl

Arguments:
    1) Path to SQLite destination database (e.g. disaster_response_db.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""

# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data_from_db(database_filepath):
    """
    Function: Load Data from DB
    
    Arguments:
        database_filepath: Path to SQLite destination database (e.g. disaster_response_db.db)
    Output:
        X: dataframe containing features
        Y: dataframe containing labels
        category_names: List of categories
    """
    
    # Read in the data from the DB
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = "messages"
    df = pd.read_sql_table(table_name,engine)
    
    # Remove child_alone column because it is all zeros
    df = df.drop(['child_alone'],axis=1)
    
    # The count of the value of 2 in the related field is negligible. Replace the positive value 2 with the positive value 1
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    # Set X & Y dataframes
    X = df['message']
    y = df.iloc[:,4:]
    
    # Get category names for later use
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Function: Tokenize
    
    Arguments:
        text: Message to tokenize
    Output:
        clean_tokens: List of clean tokens extracted from text
    """
    
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    # Remove inflection and derivation related forms of words
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    
    return clean_tokens

# Transformer to extract the first verb of text
class FirstVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Class: First Verb Extractor class
    
    Extracts the first verb of the text
    """

    # Find the first verb
    def first_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Return itself 
    def fit(self, X, y=None):
        return self

    # Transform across the X series
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.first_verb)
        return pd.DataFrame(X_tagged)

def build_pipeline():
    """
    Function: Build Pipeline
    
    Output:
        A Scikit ML Pipeline that processes text messages and applies the classifier.
        
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', FirstVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline

def evaluate_pipeline(pipeline, X_test, Y_test, category_names):
    """
    Function: Evaluate Pipeline
    
    Test the ML pipeline and show performance
    
    Arguments:
        pipeline: ML Pipeline
        X_test: Test features
        Y_test: Test labels
        category_names: category names
    """
    
    Y_pred = pipeline.predict(X_test)
    
    accuracy = (Y_pred == Y_test).mean().mean()

    print('Average accuracy {0:.2f}%'.format(accuracy*100))

    # Print the whole classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))


def save_pipeline_as_pickle(pipeline, pickle_filepath):
    """
    Function: Save Pipeline as Pickle
    
    Save the pipeline model as a pickle file
    
    Arguments:
        pipeline: Pipeline to save
        pickle_filepath: Path for .pkl file
    
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))

def main():

    # Assesses arguments and starts the ML Pipeline
    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format(database_filepath))
        X, Y, category_names = load_data_from_db(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building the pipeline ...')
        pipeline = build_pipeline()
        
        print('Training the pipeline ...')
        pipeline.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_pipeline(pipeline, X_test, Y_test, category_names)

        print('Saving pipeline to {} ...'.format(pickle_filepath))
        save_pipeline_as_pickle(pipeline, pickle_filepath)

        print('Trained model saved!')

    else:
         print("Sample Script Execution:\n\
> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl")

if __name__ == '__main__':
    main()