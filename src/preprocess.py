import os
import re
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import get_project_paths

def load_reviews(folder):
    reviews, labels = [], []
    for sentiment in ['pos', 'neg']:
        path = os.path.join(folder, sentiment)
        label = 1 if sentiment == 'pos' else 0
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding='utf-8') as f:
                text = f.read()
                reviews.append(text)
                labels.append(label)
    return reviews, labels

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

def save_numpy_data(arr, filename):
    np.save(filename, arr)
    print(f"Saved to {filename}.npy")

def load_numpy_data(filename):
    return np.load(f"{filename}.npy")

def save_tokenizer(tokenizer, path='tokenizer.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path='tokenizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def dataset_summary(train_sequences, valid_sequences, test_sequences, word_index):
    # Flatten all lengths
    train_lengths = [len(seq) for seq in train_sequences]
    valid_lengths = [len(seq) for seq in valid_sequences]
    test_lengths = [len(seq) for seq in test_sequences]
    
    print("\nDATASET SUMMARY")
    print(f"Training samples: {len(train_sequences)}")
    print(f"Valid samples:    {len(valid_sequences)}")
    print(f"Testing samples:  {len(test_sequences)}")
    print(f"Vocabulary size:  {len(word_index) + 1}")  # +1 for padding token
    
    print(f"\nAverage train review length:  {np.mean(train_lengths):.2f}")
    print(f"Std. dev. of train lengths:   {np.std(train_lengths):.2f}")
    print(f"Max train review length:      {np.max(train_lengths)}")
    print(f"Min train review length:      {np.min(train_lengths)}")
    print(f"\nAverage valid review length:  {np.mean(valid_lengths):.2f}")
    print(f"Std. dev. of valid lengths:   {np.std(valid_lengths):.2f}")
    print(f"Max valid review length:      {np.max(valid_lengths)}")
    print(f"Min valid review length:      {np.min(valid_lengths)}")
    print(f"\nAverage test review length:  {np.mean(test_lengths):.2f}")
    print(f"Std. dev. of test lengths:   {np.std(test_lengths):.2f}")
    print(f"Max test review length:      {np.max(test_lengths)}")
    print(f"Min test review length:      {np.min(test_lengths)}")

if __name__ == "__main__":
    # Get absolute path to this script's directory
    BASE_DIR, PROJECT_ROOT = get_project_paths()

    # Define data paths relative to project root
    train_folder = os.path.join(PROJECT_ROOT, 'data', 'train')
    test_folder = os.path.join(PROJECT_ROOT, 'data', 'test')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')

    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess
    train_reviews, train_labels = load_reviews(train_folder)
    test_reviews, test_labels = load_reviews(test_folder)
    train_reviews = [preprocess_text(r) for r in train_reviews]
    test_reviews = [preprocess_text(r) for r in test_reviews]
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Create validation split from train (10%)
    train_reviews, valid_reviews, train_labels, valid_labels = train_test_split(train_reviews, train_labels, test_size=0.1, random_state=42, stratify=train_labels)

    # Tokenizer
    tokenizer_path = os.path.join(output_dir, 'tokenizer.pkl')

    if os.path.exists(tokenizer_path):
        print("Tokenizer found, loading tokenizer...")
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        print("No tokenizer found, building new tokenizer...")
        tokenizer = Tokenizer(num_words=10000, oov_token="<UNK>")
        tokenizer.fit_on_texts(train_reviews)
        save_tokenizer(tokenizer, tokenizer_path)

    # Convert to sequences
    train_seqs = tokenizer.texts_to_sequences(train_reviews)
    valid_seqs = tokenizer.texts_to_sequences(valid_reviews)
    test_seqs = tokenizer.texts_to_sequences(test_reviews)

    # Obtain preprocessing statistics for report
    dataset_summary(train_seqs, valid_seqs, test_seqs, tokenizer.word_index)

    # Save data padded to different lengths
    for length in [25, 50, 100]:
        train_pad = pad_sequences(train_seqs, maxlen=length)
        valid_pad = pad_sequences(valid_seqs, maxlen=length)
        test_pad = pad_sequences(test_seqs, maxlen=length)
        save_numpy_data(train_pad, os.path.join(output_dir, f'train_pad_{length}'))
        save_numpy_data(valid_pad, os.path.join(output_dir, f'valid_pad_{length}'))
        save_numpy_data(test_pad, os.path.join(output_dir, f'test_pad_{length}'))

    # Save labels
    save_numpy_data(train_labels, os.path.join(output_dir, 'train_labels'))
    save_numpy_data(valid_labels, os.path.join(output_dir, 'valid_labels'))
    save_numpy_data(test_labels, os.path.join(output_dir, 'test_labels'))