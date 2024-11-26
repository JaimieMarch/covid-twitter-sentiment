import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from database import database_manager as dbm


def get_data(table):
    print("Retrieving data...")  
    query = f"""
        SELECT * 
        FROM "{table}" 
    """
    df = dbm.query_db(query)
    return df

def store_data(table_name, dataframe):
    print("Writing table...")
    dbm.create_table(table_name=table_name, dataframe=dataframe, replace=True)
    query = f"""
        SELECT * FROM "{table_name}"
    """
    df = dbm.query_db(query)
    print(df.head(15))


def sentiment_analysis(data):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    df_clean = data

    print("Analyzing Sentiment")
    for _, row in df_clean.iterrows():
        sentiment = analyzer.polarity_scores(row['text'])
        sentiment['tweet'] = row['text']
        sentiment_scores.append(sentiment)

    df_sentiments = pd.DataFrame(sentiment_scores)
    df_sentiments = df_sentiments.reset_index(drop=True)
    print(df_sentiments.head(15))
    return df_sentiments


def main():
    df_data = get_data("processed_twitter_data")
    df_sent = sentiment_analysis(df_data)
    store_data(table_name="sentiment_scores", dataframe=df_sent)
    

if __name__ == '__main__':
    nltk.download('vader_lexicon')
    main()