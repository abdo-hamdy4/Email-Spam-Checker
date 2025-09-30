import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['text', 'spam']] # Keep only relevant columns
    df.columns = ['text', 'label'] # Rename for consistency
    return df

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def text_preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def vectorize_data(X_train, X_test):
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized, vectorizer

def train_model(X_train_vectorized, y_train):
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vectorized, y_train)
    return model

def save_model(model, filepath):
    joblib.dump(model, filepath)

def evaluate_model(model, X_test_vectorized, y_test):
    y_pred = model.predict(X_test_vectorized)
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print("\n--- Accuracy Score ---")
    print(f"{accuracy_score(y_test, y_pred):.4f}")

    # Plotting confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    return y_pred

if __name__ == "__main__":
    df = load_data('./data/emails.csv')
    
    print("--- Preprocessing Text ---")
    df['processed_text'] = df['text'].apply(text_preprocess)
    print(df.head())
    
    X = df['processed_text']
    y = df['label']

    X_train, X_test, y_train, y_test = split_data(X, y)
    
    X_train_vectorized, X_test_vectorized, vectorizer = vectorize_data(X_train, X_test)
    
    print(f"\nTraining model on {X_train_vectorized.shape[0]} samples...")
    model = train_model(X_train_vectorized, y_train)
    print("Model training complete.")
    
    print("\n--- Evaluating Model ---")
    y_pred = evaluate_model(model, X_test_vectorized, y_test)
    
    print("\n--- Saving Model ---")
    save_model(model, './models/spam_model.pkl')
    save_model(vectorizer, './models/vectorizer.pkl')