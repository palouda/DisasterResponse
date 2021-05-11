"""
Preprocessing of Data
Project: Disaster Response Pipeline

Arguments:
    1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response_db.db)

Sample Script Syntax:

> python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite destination db>

Sample Script Execution from root directory:
> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

"""

# Import all the relevant libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
 
def load_messages_with_categories(messages_filepath, categories_filepath):
    """
    Function: Load Messages with Categories
    
    Arguments:
        messages_filepath: Path to the CSV file containing messages
        categories_filepath: Path to the CSV file containing categories
    Output:
        df: Combined data containing messages and categories
    """
    
    # Read the messages
    messages = pd.read_csv(messages_filepath)
    
    # Read the categories
    categories = pd.read_csv(categories_filepath)
    
    # Merge the datasets
    df = pd.merge(messages,categories,on='id')
    
    # Return the merged dataset
    return df 

def clean_categories_data(df):
    """
    Function: Clean Categories Data
    
    Arguments:
        df: Data containing messages and categories
    Outputs:
        df: Data containing messages and categories with clean categories
    """
    
    # Split the categories
    categories = df['categories'].str.split(pat=';',expand=True)
    
    # Set the name of the category columns to the name of the category
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    # Set the category columns to numeric 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    
    # Drop the original categories column
    # Concatenate with the new categories and remove duplicates
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df

def save_data_to_db(df, database_filename):
    """
    Function: Save Data to DB
    
    Arguments:
        df: Cleaned data containing messages and categories
        database_filename: Path to SQLite destination database
    """
    
    # Save data to the "messages" table in the DB
    # Replace the table if it already exists
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')

def main():
    """
    Function: Main
    
    Arguments:
        <path to messages csv file> <path to categories csv file> <path to sqllite destination db>
        
    Tasks:
        1) Load messages and categories
        2) Clean categories
        3) Save data to DB
    """
    
    # Execute the ETL pipeline if the count of arguments is matching to 4
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] # Extract the parameters in relevant variable

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))
        
        df = load_messages_with_categories(messages_filepath, categories_filepath)

        print('Cleaning categories data ...')
        df = clean_categories_data(df)
        
        print('Saving data to SQLite DB : {}'.format(database_filepath))
        save_data_to_db(df, database_filepath)
        
        print('Cleaned data has been saved to database!')
    
    else: # Print the help message so that user can execute the script with correct parameters
        print("Sample Script Execution:\n\
> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db \n\
Arguments Description: \n\
1) Path to the CSV file containing messages (e.g. data/disaster_messages.csv)\n\
2) Path to the CSV file containing categories (e.g. data/disaster_categories.csv)\n\
3) Path to SQLite destination database (e.g. data/DisasterResponse.db)")

if __name__ == '__main__':
    main()