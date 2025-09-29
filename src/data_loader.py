import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import string
import nltk

def load_and_preprocess_data():
    """Load and preprocess the movie review dataset.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with reviews and encoded sentiments.
    """

    # Load the dataset
    df = pd.read_csv("../data/Movie_Dataset.csv")
    print(df.head())

    print(df.columns.unique())

    # One-Hot Encoding for sentiment column
    label_encoder = LabelEncoder()
    sentiment_encoded = label_encoder.fit_transform(df[['sentiment']])
    df['sentiment'] = sentiment_encoded
    print(df.head())

    # Sampling 10% of the dataset for quicker processing
    df_train = df.sample(frac=0.1, random_state=42)

    # Balancing the dataset
    df = df.groupby('sentiment').apply(lambda x: x.sample(
            df_train['sentiment'].value_counts().min(), random_state=42)).reset_index(drop=True)

    # Checking class distribution
    print(df['sentiment'].value_counts())

    # Text Preprocessing Functions
    def remove_between_square_brackets(text):
        return re.sub(r'\[.*?\]', '', text)

    # Remove punctuation
    def remove_punctuation(text):
        punctuation = list(string.punctuation)
        for p in punctuation:
            text = text.replace(p, '')
        return text

    # Remove special characters
    def remove_special_characters(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Process text: remove URLs, emails, HTML entities, and apply stemming
    def process_text(text):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'www\S+', '', text)   # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
        text = re.sub(r'&quot;', '', text)
        text = re.sub(r'<br /><br />', '', text)
        stemmer = nltk.stem.SnowballStemmer("english")
        text = ' '.join([stemmer.stem(word) for word in text.split()])
        pattern = r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text.strip().lower()

    # Combined function to clean text
    def removing_the_noise(text):
        text = remove_between_square_brackets(text)
        text = remove_punctuation(text)
        text = remove_special_characters(text)
        text = process_text(text)
        return text

    df['review'] = df['review'].apply(removing_the_noise)
    return df


df = load_and_preprocess_data()

# Block to prevent code from running on import
if __name__ == "__main__":
    print("--- Running data_loader.py directly ---")
    print("Dataframe header:")
    print(df.head())
    print("\nClass Distribution:")
    print(df['sentiment'].value_counts())
    print("\nData loading and preprocessing complete.")   