import pandas as pd
import streamlit as st


from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from database import database_manager as dbm


# queries database to get the table data
def get_sentiment_data(table):
    print(f"Retrieving {table} table...")
    
    # Tables in DB:
    # table = "raw_twitter_data"
    # table = "processed_twitter_data"
    # table = "sentiment_scores"
    
    query = f"""
        SELECT * 
        FROM "{table}" 
    """
    df = dbm.query_db(query)
    return df


# run: streamlit run streamlit.py
def main():
    df_sent = get_sentiment_data("sentiment_scores")
    df_proc = get_sentiment_data("processed_twitter_data")
    df = pd.merge(df_sent, df_proc, on='id', how='inner')
    print(df.head(15))

    st.write("### Sentiment Scores DataFrame")
    st.dataframe(df_sent.head(100))

    st.write("### Processed Twitter DataFrame")
    st.dataframe(df_proc.head(100))

    st.write("### Merged DataFrame")
    st.dataframe(df.head(100))

    df_temp = df_sent[['neg', 'pos']]
    st.bar_chart(df_temp.head(50))

    



if __name__ == '__main__':
    main()