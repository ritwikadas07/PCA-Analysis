# PCA-Analysis

## Purpose

The purpose of this code is to read from a CSV file, understand the context of each line, and plot the eigenvalues using PCA, focusing on the relevance of the target words "Cats," "Dogs," and "Computers."

## Description

The process involves tokenizing and lemmatizing the text data using NLTK, followed by vectorizing the processed text using either the TF-IDF (Term Frequency-Inverse Document Frequency) method or word embeddings via the GloVe model. TF-IDF calculates the importance of each word within the corpus, while GloVe embeddings provide vectors that capture semantic relationships between words. 
Principal Component Analysis (PCA) is applied to reduce the dimensionality of these vectors to three components, and the results are visualized in a 3D scatter plot, showing the relationships and relevance of the target words within the context of the text data.

I have first tried to create the code using TD-IDF Vectorization. To improve the code i changed to Word Embedding Vectorization.

## Difference in Vectorization Methods

The primary difference between the two codes lies in the vectorization method used for representing the text data:

### TF-IDF Vectorization

- Converts the processed text data into numerical vectors based on the importance of each word within the corpus.
- Vectors are sparse and high-dimensional, with each word represented by its importance.

## Previous Result with Vectorization Method used - TD-IDF

<p align="center">
<img src="/images/image1.png" "Packet Structure">
</p>


### Word Embeddings

- Uses pre-trained GloVe embeddings to represent words as dense, low-dimensional vectors.
- Captures semantic relationships between words, with similar words located close to each other in the vector space.

## Result with Vectorization Method used - TD-IDF

<p align="center">
<img src="/images/image2.png" "Packet Structure">
</p>

## Purpose and Advantage of Word Embeddings

Word embeddings serve a better purpose in this case because they capture semantic relationships between words. Words that are semantically similar (e.g., "cats" and "dogs") have similar embeddings and, therefore, close eigenvalues when PCA is applied. This semantic proximity is beneficial for understanding the contextual relevance and relationships between words in the text data.

## Decision Rationale

Due to the advantages of word embeddings in capturing semantic meaning, the approach using GloVe embeddings was chosen. This method ensures that semantically related words are positioned closely in the vector space, leading to more meaningful PCA visualizations and better insights into the text data. By contrast, TF-IDF focuses on word frequency and importance but does not account for semantic relationships, which can limit the analysis, especially for tasks requiring an understanding of word meanings and contexts.
