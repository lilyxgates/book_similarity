# Similarity Assessment for Book Recommendation System
*Written by Lily Gates*
* March 2025*

## Description
This project explores similarity assessment techniques in data science, with a particular focus on comparing books based on their content. By utilizing methods like **cosine similarity** and **Euclidean distance**, this project enables a bookstore manager to optimize book displays, recommendations, and inventory based on content or genre similarity. The ultimate goal is to provide actionable insights that can be easily interpreted by stakeholders to make data-driven decisions.

## Usage
To run the similarity analysis:

1. Use a dataset of books that includes relevant features such as **book descriptions**, **subject categories**, and **keywords**.
2. The Python script allows you to query books by selecting a specific book, and it will return a ranked list of the most similar books.
3. The output can be used to:
   - Reorganize in-store displays based on similar genres or content.
   - Generate book recommendation lists for customers.
   - Inform inventory decisions, helping to understand customer preferences based on book similarities.

## Required Dependencies
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `scipy`: For scientific and technical computing (used for calculating Euclidean distance).
- `scikit-learn`: For implementing machine learning algorithms, particularly cosine similarity.
- `Google Books API`: For data enrichment and retrieving book information.

## Example Usage
```bash
python book_similarity.py --input_books books_data.csv --query_book "Harry Potter and the Sorcerer's Stone"
