import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('wordnet')

def process_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]

def calculate_distances(points):
    num_points = points.shape[0]
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distances[i, j] = np.linalg.norm(points[i] - points[j])
            distances[j, i] = distances[i, j]
    return distances

def main():
    st.title("Word Embeddings Visualization")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        processed_corpus = [process_text(text) for text in df['text']]
        
        try:
            word2vec_model = Word2Vec(sentences=processed_corpus, vector_size=300, window=10, min_count=1, workers=4, epochs=200)
        except Exception as e:
            st.error(f"Error training Word2Vec model: {e}")
            return
        
        target_words = st.text_input("Enter target words (comma-separated):")
        target_words = [word.strip() for word in target_words.split(',')]
        
        word_embeddings = []
        for word in target_words:
            if word in word2vec_model.wv:
                word_embeddings.append(word2vec_model.wv[word])
            else:
                st.warning(f"Word '{word}' not found in vocabulary.")
        word_embeddings = np.array(word_embeddings)
        
        # Normalize word embeddings
        st.write("Shape of word embeddings before normalization:", word_embeddings.shape)
        if len(word_embeddings.shape) == 1:
            word_embeddings = word_embeddings.reshape(-1, 1)
        word_embeddings /= np.linalg.norm(word_embeddings, axis=1, keepdims=True)
        st.write("Shape of word embeddings after normalization:", word_embeddings.shape)

        
        pca = PCA(n_components=3)
        try:
            pca_result = pca.fit_transform(word_embeddings)
        except Exception as e:
            st.error(f"Error performing PCA: {e}")
            return
        
        distances = calculate_distances(pca_result)
        st.subheader("Distances between points:")
        for i, word1 in enumerate(target_words):
            for j, word2 in enumerate(target_words):
                if i < j:
                    st.write(f"Distance between {word1} and {word2}: {distances[i, j]:.4f}")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for idx, word in enumerate(target_words):
            ax.scatter(pca_result[idx, 0], pca_result[idx, 1], pca_result[idx, 2], color='blue')
            ax.text(pca_result[idx, 0], pca_result[idx, 1], pca_result[idx, 2], word, color='red', fontsize=12)
        
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_zlabel('PCA3')
        ax.set_title('3D PCA of Word Embeddings')
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()
