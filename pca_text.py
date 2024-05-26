import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
import zipfile
import os 
from tqdm import tqdm

nltk.download('punkt')
nltk.download('wordnet')

def download_and_unzip(url, file_to_extract):
    # Extract file name from URL
    filename = url.split('/')[-1]
    
    # Make a GET request to the URL, stream the response
    print("Downloading GloVe Vectors file")
    response = requests.get(url, stream=True)
    print("Finished downloading GloVe Vectors file")
    
    # Get the total file size in bytes
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    # Initialize the progress bar
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    # Download and write data to file
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    
    # Close the progress bar
    progress_bar.close()
    
    # Unzip the file
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extract(file_to_extract)
    
    # Delete the zip file
    os.remove(filename)

def process_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def main():
    download_and_unzip("https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip", "glove.6B.300d.txt")
    
    csv_file_path = "sample_text_data.csv"
    df = pd.read_csv(csv_file_path)

    processed_corpus = [process_text(text) for text in df['text']]
    
    glove_path = 'glove.6B.300d.txt'  # Adjust the path as necessary
    embeddings_index = load_glove_embeddings(glove_path)

    # Get embeddings for specific words
    target_words = ['cats', 'dogs', 'computers']
    word_embeddings = np.array([embeddings_index[word] for word in target_words if word in embeddings_index])

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(word_embeddings)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for idx, word in enumerate(target_words):
        if word in embeddings_index:
            ax.scatter(pca_result[idx, 0], pca_result[idx, 1], pca_result[idx, 2], color='blue')
            ax.text(pca_result[idx, 0], pca_result[idx, 1], pca_result[idx, 2], word, color='red', fontsize=12)
            
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.set_title('3D PCA of Word Embeddings')
    
    plt.show()

if __name__ == "__main__":
    main()