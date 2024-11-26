import pandas as pd

from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from database import database_manager as dbm

'''
queries database to get the sentiment scores
returns df
'''
def get_sentiment_data():
    print("Retrieving sentiment scores table...")
    '''
        other tables in DB:
        table = "raw_twitter_data"
        table = "processed_twitter_data"
    '''
    table = "sentiment_scores"
    query = f"""
        SELECT * 
        FROM "{table}" 
    """
    df = dbm.query_db(query)
    return df


def main():
    df_sent = get_sentiment_data()
    print(df_sent.head(15))



if __name__ == '__main__':
    main()