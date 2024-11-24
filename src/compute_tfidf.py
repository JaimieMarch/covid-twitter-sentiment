import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd

# Load environment variables
load_dotenv()

# Database connection URL from .env
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise ValueError("DB_URL not found. Ensure it's set in the .env file.")

# Create database engine
engine = create_engine(DB_URL)

# Function to clear the table
def clear_tfidf_results_table():
    print("Clearing the contents of `tfidf_results_1_2` table...")
    query = text("TRUNCATE TABLE tfidf_results_1_2")
    with engine.connect() as conn:
        conn.execute(query)
    print("Table `tfidf_results_1_2` has been cleared.")

# Function to compute and store top terms
def compute_and_store_top_terms():
    print("Fetching data from `tfidf_results_1`...")
    
    # Fetch data from tfidf_results_1
    query = text("SELECT term, tfidf_score FROM tfidf_results_1")
    with engine.connect() as conn:
        data = pd.read_sql(query, conn)
    
    print("Computing aggregate scores...")
    
    # Aggregate data
    aggregated = (
        data.groupby("term")
        .agg(
            total_weight=("tfidf_score", "sum"),
            avg_weight=("tfidf_score", "mean"),
            tweet_count=("tfidf_score", "size"),
        )
        .reset_index()
    )
    
    # Sort by avg_weight and limit to top 150 terms
    top_terms = aggregated.sort_values(by="avg_weight", ascending=False).head(150)
    
    print("Storing top terms into `tfidf_results_1_2` table...")
    with engine.connect() as conn:
        top_terms.to_sql(
            name="tfidf_results_1_2",
            con=conn,
            if_exists="append",
            index=False,
        )
    print("Top terms stored successfully in `tfidf_results_1_2`.")

# Function to view the results
def view_tfidf_results_1_2_ordered():
    print("Fetching data from `tfidf_results_1_2` table, ordered by `avg_weight` descending...")
    
    query = text("SELECT * FROM tfidf_results_1_2 ORDER BY avg_weight DESC")
    
    with engine.connect() as conn:
        results = pd.read_sql(query, conn)
    
    print("Displaying the top terms ordered by `avg_weight`...")
    print(results.head())  # Display the first few rows
    return results

# Entry point
if __name__ == "__main__":
    # Clear the table before repopulating
    clear_tfidf_results_table()
    
    # Compute and store top terms
    compute_and_store_top_terms()
    
    # View the stored results
    df = view_tfidf_results_1_2_ordered()









