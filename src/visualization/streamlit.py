import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
from wordcloud import WordCloud

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from database import database_manager as dbm



plt.style.use('dark_background')


def apply_custom_style():
    plt.style.use('dark_background') 
    sns.set_palette(['#FF69B4', '#1E90FF', '#32CD32', '#FF1493', '#8A2BE2'])

    plt.rcParams.update({
        'axes.facecolor': '#212121',  
        'figure.facecolor': '#212121',  
        'axes.edgecolor': 'white',  
        'grid.alpha': 0.3,  
        'grid.color': 'white', 
        'grid.linestyle': '--', 
        'axes.labelsize': 14, 
        'axes.labelcolor': 'white',  
        'xtick.labelsize': 12,  
        'ytick.labelsize': 12,  
        'xtick.color': 'white', 
        'ytick.color': 'white',  
        'axes.titlesize': 16, 
        'axes.titleweight': 'bold',
        'axes.titlecolor': 'white',  
        'figure.figsize': (10, 6), 
        'figure.dpi': 100 
    })

# queries database to get the table data
@st.cache_data
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

# Classifies rows into sentiments based on compound score (overall polarity)
def classify_sentiment(row):
    compound = row['compound']
    
    if compound > 0.05:
        return 'positive'
    elif compound < -0.05:
        return 'negative'
    else:
        return 'neutral'

# Creates a time series plot
def create_time_series(df_clean):

    df_clean['date'] = pd.to_datetime(df_clean['date'])  

    df_clean = df_clean[df_clean['sentiment'] != 'neutral']

    sentiment_trend = df_clean.groupby([df_clean['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)

    # Plotting the sentiment trend
    sentiment_trend.plot(kind='line', figsize=(10, 6))
    plt.title('Sentiment Trend (Negative vs Positive)')
    plt.xlabel('Date')
    plt.ylabel('Count of Tweets')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment', labels=['Negative','Positive'])
    plt.tight_layout()

    st.pyplot(plt)

def create_sentiment_summary(df_sent):
    # Get counts of all sentiments
    positive_count = df_sent[df_sent['sentiment'] == 'positive'].shape[0]
    negative_count = df_sent[df_sent['sentiment'] == 'negative'].shape[0]
    neutral_count = df_sent[df_sent['sentiment'] == 'neutral'].shape[0]
    
    # Get avg polarity for each group
    avg_polarity_positive = df_sent[df_sent['sentiment'] == 'positive']['compound'].mean()
    avg_polarity_negative = df_sent[df_sent['sentiment'] == 'negative']['compound'].mean()
    avg_polarity_neutral = df_sent[df_sent['sentiment'] == 'neutral']['compound'].mean()
    

    st.markdown("### Sentiment Summary")
    st.write("The sentiment summary provides an overview of the sentiment distribution in the dataset.")
    st.write("Included are the counts of each classification of tweet, as well as the average polarity scores for each sentiment group.")

    st.metric("Positive Tweets", positive_count)
    st.metric("Negative Tweets", negative_count)
    st.metric("Neutral Tweets", neutral_count)
 
    st.markdown("### Average Polarity")
    st.write(f"**Positive Sentiment Average Polarity** {avg_polarity_positive:.2f}")
    st.write(f"**Negative Sentiment Average Polarity** {avg_polarity_negative:.2f}")
    st.write(f"**Neutral Sentiment Average Polarity** {avg_polarity_neutral:.2f}")

    sentiment_counts = df_sent['sentiment'].value_counts()
    
    # Create a bar chart
    st.write("This bar chart shows the distribution of sentiment scores in the dataset.")
    sentiment_distribution = sentiment_counts.plot(kind='bar', figsize=(8, 5), color=['#1E90FF', '#FF69B4', '#32CD32'])
    sentiment_distribution.set_title('Sentiment Frequency')
    sentiment_distribution.set_ylabel('Count of Tweets')
    sentiment_distribution.set_xlabel('Sentiment')
    st.pyplot(plt)


def create_sentiment_histogram(df_sent):
    # Create a histogram of the sentiment scores
    plt.figure(figsize=(10, 6))
    plt.hist(df_sent['compound'], bins=30, edgecolor='black', color='#8A2BE2')
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Sentiment Score (Compound)')
    plt.ylabel('Frequency')
    plt.tight_layout()

    st.pyplot(plt)

    
def tf_idf_plots(tf_idf_df, df_sent):
   

    df_sent = df_sent[df_sent['sentiment'] != 'neutral']
    tf_idf_df = tf_idf_df[tf_idf_df["tfidf_score"] > 0.1]

    no_terms = ["covidvaccine", "covid19", "covid", "vaccine", "coronavirus", "covid-19", "vaccines", "vaccination", "virus", "get", "shot", "slot"]

    tf_idf_df = tf_idf_df[~tf_idf_df["term"].isin(no_terms)]

    tf_idf_df = tf_idf_df.groupby("tweet_id").first().reset_index()
    
    df_merged = pd.merge(df_sent, tf_idf_df, left_on='id', right_on='tweet_id', how='inner')

    # Create a bar plot with most common terms for positive and negative tweets
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    df_merged[df_merged['sentiment'] == 'positive']['term'].value_counts().head(10).plot(kind='bar', ax=ax[0], color='#1E90FF')
    ax[0].set_title('Top Terms in Positive Tweets')
    ax[0].set_xlabel('Term')
    ax[0].set_ylabel('Frequency')
    df_merged[df_merged['sentiment'] == 'negative']['term'].value_counts().head(10).plot(kind='bar', ax=ax[1], color='#FF69B4')
    ax[1].set_title('Top Terms in Negative Tweets')
    ax[1].set_xlabel('Term')
    ax[1].set_ylabel('Frequency')
    plt.tight_layout()

    st.pyplot(plt)



def plot_top_tfidf_words(tf_idf_df, top_word_count):
    tf_idf_df = tf_idf_df.dropna()
    tf_idf_df = tf_idf_df[tf_idf_df["term"] != "covidvaccine"]
    tf_idf_df = tf_idf_df.sort_values(by='tfidf_score', ascending=False)

    top_terms = tf_idf_df.drop_duplicates(subset='term').head(top_word_count)

    # Create a bar plot of the top N words by frequency
    plt.figure(figsize=(12, 8))
    plt.barh(top_terms['term'], top_terms['tfidf_score'])
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Term')
    plt.title(f'Top {top_word_count} Words by TF-IDF Score')
    plt.tight_layout()


    # Show the plot
    st.pyplot(plt)

    
def word_cloud_plot(tf_idf_df, df_sent):
        # Filter out neutral sentiment tweets
    df_sent = df_sent[df_sent['sentiment'] != 'neutral']

  
    tf_idf_df = tf_idf_df[tf_idf_df["tfidf_score"] > 0.1]

    no_terms = ["covidvaccine", "covid19", "covid", "vaccine", "coronavirus", "covid-19", "vaccines", "vaccination", "virus", "get", "shot", "slot"]

    # Filter out the rows where the 'term' column matches any of the terms in the exclude_terms list
    tf_idf_df = tf_idf_df[~tf_idf_df["term"].isin(no_terms)]

    tf_idf_df = tf_idf_df.join(df_sent.set_index('id'), on='tweet_id')

    # Plot word cloud per sentiment
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for i, sentiment in enumerate(['positive', 'negative']):
        text = ' '.join(tf_idf_df[tf_idf_df['sentiment'] == sentiment]['term'])
        wordcloud = WordCloud(width=800, height=400, background_color ='black').generate(text)
        ax[i].imshow(wordcloud)
        ax[i].set_title(f'Word Cloud for {sentiment.capitalize()} Sentiment')
        ax[i].axis('off')
    plt.tight_layout()
    st.pyplot(plt)


def top_hashtags_per_sentiment(df_sent, sentiment):
    df_sent = df_sent[df_sent['sentiment'] == sentiment]

    # Make all hashtags lowercase
    df_sent['hashtags'] = df_sent['hashtags'].str.lower()

    no_terms = ["covidvaccine", "covid19", "covid", "vaccine", "coronavirus", "covid-19", "vaccines", "vaccination", "virus", "get", "shot", "slot"]



    # Flatten the list of hashtags
    hashtags = df_sent['hashtags'].str.split().explode()

    hashtags = hashtags.str.replace('[', '', regex=False).str.replace(']', '', regex=False)

    # Filter to not include the no_terms
    hashtags = hashtags[~hashtags.isin(no_terms)]

    hashtags = hashtags[hashtags != 'nan']

    hashtags = hashtags.str.strip()

    # Count the frequency of each hashtag
    hashtag_counts = hashtags.value_counts()

    # Plot the top 10 hashtags
    plt.figure(figsize=(10, 6))
    hashtag_counts.head(10).plot(kind='bar', color='#1E90FF')
    plt.title(f'Top 10 Hashtags in {sentiment.capitalize()} Tweets')
    plt.xlabel('Hashtag')
    plt.ylabel('Frequency')
    plt.tight_layout()

    st.pyplot(plt)


def follower_amount_sentiment(df_sent):
    df_sent = df_sent[df_sent['sentiment'] != 'neutral']

    # Create a scatter plot of the number of followers vs sentiment score
    plt.figure(figsize=(10, 6))
    plt.scatter(df_sent['user_followers'], df_sent['compound'], alpha=0.5)
    plt.title('Number of Followers vs Sentiment Score')
    plt.xlabel('Number of Followers')
    plt.ylabel('Sentiment Score (Compound)')
    plt.tight_layout()

    st.pyplot(plt)

def friends_amount_sentiment(df_sent):
    df_sent = df_sent[df_sent['sentiment'] != 'neutral']

    # Create a scatter plot of the number of friends vs sentiment score
    plt.figure(figsize=(10, 6))
    plt.scatter(df_sent['user_friends'], df_sent['compound'], alpha=0.5)
    plt.title('Number of Friends vs Sentiment Score')
    plt.xlabel('Number of Friends')
    plt.ylabel('Sentiment Score (Compound)')
    plt.tight_layout()

    st.pyplot(plt)

def average_followers_sentiment(df_sent):
    df_sent = df_sent[df_sent['sentiment'] != 'neutral']

    #Check for outliers 
    q1 = df_sent['user_followers'].quantile(0.25)
    q3 = df_sent['user_followers'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    #filter to remove outliers
    df_sent = df_sent[(df_sent['user_followers'] > lower_bound) & (df_sent['user_followers'] < upper_bound)]

    # Create a visual showing positive and negative sentiment and their average followers:
    avg_followers = df_sent.groupby('sentiment')['user_followers'].mean()

    plt.figure(figsize=(10, 6))
    avg_followers.plot(kind='bar', color=['#1E90FF', '#FF69B4'])
    plt.title('Average Followers by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Average Followers')
    plt.tight_layout()

    st.pyplot(plt)

def percentage_verfied_user_by_sentiment(df_sent):
    df_sent = df_sent[df_sent['sentiment'] != 'neutral']


    df_sent['user_verified'] = df_sent['user_verified'].map({'false': False, 'true': True})

    # Create a visual showing the percentage of verified users by sentiment (true or false)

    verified_users = df_sent.groupby('sentiment')['user_verified'].value_counts(normalize=True).unstack().fillna(0)



    plt.figure(figsize=(10, 6))
    verified_users.plot(kind='bar', color=['#1E90FF', '#FF69B4'])
    plt.title('Percentage of Verified Users by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Percentage of Verified Users')
    plt.tight_layout()

    st.pyplot(plt)



# run: streamlit run streamlit.py
def main():
    apply_custom_style()
    df_sent = get_sentiment_data("sentiment_scores_df")
    
      # Apply sentiment classification (cached)
    df_sent['sentiment'] = df_sent.apply(classify_sentiment, axis=1)
    df_tf_idf = get_sentiment_data("tfidf_results_1")


    #Webpage title
    st.title("Covid-19 Tweets Sentiment Analysis 🦠")

    # Desc of use case
    st.markdown("This dashboard features a sentiment analysis of tweets tagged with #Covid19, spanning from 2020 to 2022. Serving to visualize trends and insights, providing an interactive way to explore public sentiment during the pandemic. Dive into the data and discover how perceptions have evolved over time!")

    # Display the first few rows of the sentiment scores DataFrame
    st.write("Here is an example of the processed sentiment scores data:")
    st.write("### Sentiment Scores DataFrame")
    st.dataframe(df_sent.head(5))

    st.write("The sentiment scores are classified into three categories: positive, negative, and neutral. The compound score is a normalized score ranging from -1 (most negative) to 1 (most positive). The additionl sentiment column is based on the compound score.")
    
    create_sentiment_summary(df_sent)

    st.write("### Sentiment Time Series")
    st.write("The sentiment time series shows the trend of positive and negative sentiment over time.")
    create_time_series(df_sent)

    st.write("### Sentiment Histogram")
    st.write("The sentiment histogram displays the distribution of sentiment scores in the dataset. With 0 being neutral, negative scores below 0, and positive scores above 0.")
    create_sentiment_histogram(df_sent)

    st.write("### TOP TF-IDF Words")
    st.write("The bar plot below shows the top TF-IDF words in the dataset. Discluding the term 'covidvaccine' as it is the most common term.")
    plot_top_tfidf_words(df_tf_idf, 25)

    st.write("### Most common terms in positive and negative sentiments")
    st.write("The bar plots below show the most common terms in positive and negative tweets.")
    st.write("Note, obvious terms, such as 'covid19', 'vaccine', etc., have been excluded from the analysis. In an attempt to identify more insightful terms.")
    tf_idf_plots(df_tf_idf, df_sent)

    st.write("### Word Cloud")
    st.write("The word cloud displays the most common terms in the tweets for each sentiment category.")
    word_cloud_plot(df_tf_idf, df_sent)

    conspiracy_count = df_tf_idf[df_tf_idf['term'].str.lower() == 'conspiracy'].shape[0]
    st.write(f"Number of tweets mentioning 'conspiracy': {conspiracy_count}")


    st.write("### Top Hashtags")
    st.write("The bar plots below show the top 10 hashtags in positive and negative tweets.")
    top_hashtags_per_sentiment(df_sent, 'positive')

    st.write("### Followers vs Sentiment")
    st.write("The scatter plot below shows the relationship between the number of followers and sentiment score.")
    follower_amount_sentiment(df_sent)

    st.write("### Verified Users vs Sentiment")
    st.write("The plot below shows the average amount of followers per sentiment (outliers have been omitted by using the IQR).")
    average_followers_sentiment(df_sent)

    st.write("### Friends vs Sentiment")
    st.write("The scatter plot below shows the relationship between the number of friends and sentiment score.")
    friends_amount_sentiment(df_sent)

    st.write("### Average Followers by Sentiment")
    st.write("The bar plot below shows the average number of followers by sentiment.")
    percentage_verfied_user_by_sentiment(df_sent)
    



if __name__ == '__main__':
    main()