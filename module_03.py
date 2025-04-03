# Lily Gates
# Module 3

import pandas as pd
import json
import os
import yaml  # For holding my API key
import requests
import time  # For sleep requests
import urllib.parse  # For dealing with spaces in book titles search
import scipy.spatial.distance
from sklearn.feature_extraction.text import TfidfVectorizer  # Analyze text as numeric data to calculate similarity


######################################
### GET FULL PATH FILE AND API KEY ###
######################################

# Current Directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# File with APIs
api_keys = "keys.yml"
api_keys_path = os.path.join(current_dir, api_keys)

# Read in API Key for Google Books API
with open(api_keys_path, 'r') as file:
    keys = yaml.safe_load(file)

google_books_key = keys['google_books_key']

# Confirm API key for Google Books exists
if not google_books_key:
    raise ValueError("Google Books API key is missing! Check keys.yml file.")

print("\n###########################################################\n")
print(f"Confirmed -- Google Books API key: {google_books_key}")
print("\n###########################################################\n")

########################
### READ IN API DATA ###
########################

# List of books you want to query
target_books = ["Harry Potter and the Chamber of Secrets",
                "To Kill a Mockingbird",
                "The Great Gatsby"]


def get_books_from_api(book_title, google_books_key, max_results=10):
    """
    Queries the Google Books API for books related to the given title.
    
    Args:
        book_title (str): The title of the book to search for.
        google_books_key (str): The Google Books API key.
        max_results (int): Number of books to retrieve (default 10).
    
    Returns:
        list: A list of dictionaries containing book details.
    """
    
    encoded_title = urllib.parse.quote(book_title)  # Encode title for URL
    url = f'https://www.googleapis.com/books/v1/volumes?q={encoded_title}&maxResults={max_results}&key={google_books_key}'
    
    response = requests.get(url)
    
    # Debugging if Error with API Request
    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        return None
    
    data = response.json()

    # Saving Results in `books_list` (list of dictionaries)
    books_list = []
    if 'items' in data:
        for item in data['items']:
            book_info = item.get('volumeInfo', {})
            title = book_info.get('title', 'No Title')
            authors = book_info.get('authors', ['No Author'])
            genres = book_info.get('categories', ['No Genre'])
            description = book_info.get('description', 'No Description')

            books_list.append({
                'Title': title,
                'Authors': ', '.join(authors),
                'Genres': ', '.join(genres),
                'Description': description
            })

    # Add a 0.3-second delay between requests
    time.sleep(0.3)  # This will delay each API request by 0.3 seconds
    
    return books_list

# Query API for all books in the list with a delay of 0.3 seconds between requests
all_books = []
for book in target_books:
    books_data = get_books_from_api(book, google_books_key, max_results=10)
    if books_data is None:  # Check if the API request failed
        print(f"Skipping '{book}' due to an error.")
    else:
        all_books.extend(books_data)
    time.sleep(0.3)  # Add a 0.3-second delay between requests

# Convert to DataFrame
df_books = pd.DataFrame(all_books)

# Debug: Check if DataFrame is empty
if df_books.empty:
    print("Error: The DataFrame is empty. No data was collected from the API.")
    exit()
else:
    print(f"Success: Collected {len(df_books)} books from the API.")

print("\nCollected book data:\n", df_books.head())

###############################
### COMPUTE TEXT SIMILARITY ###
###############################

# Fill missing descriptions with an empty string
df_books['Description'] = df_books['Description'].fillna('')

# TF-IDF Vectorization for book descriptions
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_books['Description'])

# Compute Cosine Similarity
cos_sim_matrix = 1 - scipy.spatial.distance.cdist(X.toarray(), X.toarray(), metric='cosine')

# Convert to DataFrame for easier analysis
similarity_df = pd.DataFrame(cos_sim_matrix, index=df_books['Title'], columns=df_books['Title'])

######################################
### FIND TOP 10 MOST SIMILAR BOOKS ###
######################################

def get_top_similar_books(target_book, df, book_df, top_n=10):
    """
    Finds the top N most similar books to a given book, excluding the same author.

    Args:
        target_book (str): The book title to find similarities for.
        df (pd.DataFrame): The cosine similarity DataFrame.
        book_df (pd.DataFrame): DataFrame containing book details.
        top_n (int): Number of top similar books to retrieve.

    Returns:
        pd.DataFrame: A DataFrame of top similar books.
    """
    if target_book not in df.index:  # This should check against similarity_df.index
        print(f"Book '{target_book}' not found in the similarity matrix.")
        return pd.DataFrame()
    
    # Get the author(s) of the target book
    target_authors = book_df.loc[book_df['Title'] == target_book, 'Authors'].values[0]

    # Get similarity scores for the target book, sort in descending order
    similar_books = df[target_book].sort_values(ascending=False)

    # Filter out the target book itself and books by the same author
    filtered_books = [
        book for book in similar_books.index
        if book != target_book and book_df.loc[book_df['Title'] == book, 'Authors'].values[0] != target_authors
    ]

    # Return the top N most similar books
    return book_df[book_df['Title'].isin(filtered_books[:top_n])]

# Print results for each target book
for target in target_books:
    print(f"\nTop 10 Books Similar to '{target}' (Excluding Same Author):")
    similar_books = get_top_similar_books(target, similarity_df, df_books)

    if not similar_books.empty:
        # Iterate through the DataFrame rows and print the top similar books and their similarity scores
        for rank, (index, row) in enumerate(similar_books.iterrows(), start=1):
            title = row['Title']
            similarity_score = row['Similarity Score']  # Assuming you have a column for similarity score
            print(f"{rank}. {title} (Similarity Score: {similarity_score:.4f})")
    else:
        print("No similar books found.")
