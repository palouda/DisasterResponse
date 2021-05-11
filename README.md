# Disaster Response Pipeline Project

## Table of Contents
1. [Introduction](#introduction)
2. [Execution](#execution)
	1. [Dependencies](#dependencies)
	2. [Installing](#installation)
	3. [Interface](#interface)
	4. [Files](#files)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)

<a name="introduction"></a>
## Introduction

Figure Eight processes messages in real time for response to disaster events. The datasets for this project contain messages and tweets from actual disasters that have been sorted and labelled. The goal of this project is to build a natural language processing model that accurately categorizes new messages.

This project is divided in the following key sections:

1. Data Collection and Processing 
    - building an ETL pipeline to extract data from the tweets and messages, as well as the categorization of those tweets and messages, then cleaning the data and storing it in a database for further use.
2. Data Modelling 
    - building and training an ML (Machine Learning) Pipeline to categorize the messages from the ETL.
3. User Interface 
    - build a web app to take new messages and process them and categorize them 

<a name="execution"></a>
## Execution

<a name="dependencies"></a>
### Dependencies
* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

<a name="installation"></a>
### Installing
To clone the project:
```
git clone https://github.com/palouda/DisasterResponse.git
```
<a name="interface"></a>
### Interface
1. Load and Train the Model

    - Open a terminal in the project root directory
    - Run the ETL Pipeline to load and clean the given data and build the database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - Run the ML pipeline to train the classifier and build the pickle file
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the web application to classify new messages
    `python app/run.py`

3. Open a browser and go to http://0.0.0.0:3001/

<a name="files"></a>
### Files

In the **data** and **models** folder you can find two jupyter notebook that will help you understand how the model works step by step:
1. **data/ETL Preparation Notebook.ipynb**: The implementation of the ETL Pipeline. Loading and cleaning the data and storing it to the database
2. **models/ML Pipeline Preparation Notebook.ipynb**: The implementation of the Machine Learning Pipeline. Building and training the model.
3. **data/disaster_categories.csv**: The raw data file for the categorization of messages
4. **data/disaster_messages.csv**: The raw data file for the categorized messages
5. **data/process_data.py**: Used for data cleaning, feature extraction and storing data for the ETL Pipeline
6. **models/train_classifier.py**: The ML Pipeline for loading data, training the model and saving the model to a pickle file
7. **app/run.py**: The executable for launching the web app
8. **app/templates/go.html**: Template for the web app
9. **app/templates/master.html** Master template for the web app

<a name="authors"></a>
## Authors

* [Christian Palouda](https://github.com/palouda)

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) - provider of the outline of the project
* [Figure Eight](https://www.figure-eight.com/) - provider of the datasets 
