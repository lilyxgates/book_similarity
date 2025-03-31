# Lily Gates
# Module 3

import pandas as pd
import json
from scipy.sparse import lil_matrix
import scipy.spatial.distance
import os
import yaml
import requests

######################################
### GET FULL PATH FILE AND API KEY ###
######################################

# Display Current Directory and Read in API Key
# Assumes a file containing a dict of API key's is in the same dir as current open dir

# Current Directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# File with APIs
api_keys = "keys.yml"

# Construct full file path to API key dict
api_keys_path = os.path.join(current_dir, api_keys)

# Read in File with Current API Key for Google Books API
with open(api_keys_path, 'r') as file:
    keys = yaml.safe_load(file)

google_books_key = keys['google_books_key']

# Confirm key
print("\n###########################################################\n")
print(f"MY PERSONAL KEY: {google_books_key}")
print("\n###########################################################\n")


########################
### READ IN API DATA ###
########################

# List of books you want to query
books = ["Harry Potter and the Chamber of Secrets", "To Kill a Mockingbird", "The Great Gatsby"]

# Function to query Google Books API
def get_book_data(book_title):
    url = f'https://www.googleapis.com/books/v1/volumes?q={book_title}&key={google_books_key}'
    response = requests.get(url)
    data = response.json()

    # Debugging: Check if we get a response from the API
    print(f"API response for '{book_title}':", data)

    # Extract relevant information if the response contains items
    if 'items' in data:
        book_info = data['items'][0]['volumeInfo']
        title = book_info.get('title', 'No Title')
        authors = book_info.get('authors', ['No Author'])
        genres = book_info.get('categories', ['No Genre'])
        description = book_info.get('description', 'No Description')
        
        return {
            'Title': title,
            'Authors': ', '.join(authors),
            'Genres': ', '.join(genres),
            'Description': description
        }
    else:
        print(f"No items found for '{book_title}'")
        return None

# Query books and store results in a list
book_data = []
for book in books:
    book_info = get_book_data(book)
    if book_info:
        book_data.append(book_info)

# Create DataFrame
df_books = pd.DataFrame(book_data)

# Debugging: Check if DataFrame is empty
if df_books.empty:
    print("The DataFrame is empty. No data was collected from the API.")
else:
    print("Collected book data:", df_books)

# Proceed with similarity calculation if DataFrame is not empty
if not df_books.empty:
    # Fill missing descriptions with empty string
    df_books['Description'] = df_books['Description'].fillna('')

    # Simple text vectorization (using CountVectorizer)
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df_books['Description'])

    # Cosine Similarity
    cos_distances = scipy.spatial.distance.cdist(X.toarray(), X.toarray(), metric='cosine')

    # Display Cosine Similarity matrix
    print("\nCosine Similarity Matrix:\n", cos_distances)

    # If you want to compare each pair of books
    for i, book_1 in enumerate(books):
        for j, book_2 in enumerate(books):
            if i != j:
                similarity_score = 1 - cos_distances[i][j]
                print(f"\nSimilarity between '{book_1}' and '{book_2}': {similarity_score:.4f}")
