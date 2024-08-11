# app/preprocess.py
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pymongo
from urllib.parse import quote_plus

def fetch_data():
    # Fetch data from MongoDB
    # Encode password for MongoDB URI
    username = 'utkarsh'
    password = 'PRO@rishit2702'
    encoded_password = quote_plus(password)
    connection_string = f'mongodb+srv://{username}:{encoded_password}@sentimentanalysiscluste.ysle9.mongodb.net/?retryWrites=true&w=majority&appName=SentimentAnalysisCluster'
    
    # Connect to MongoDB and fetch your data
    client = pymongo.MongoClient(connection_string)
    db = client['sentiment_analysis_db']
    collection = db['reviews']
    data = pd.DataFrame(list(collection.find()))
    return data

def preprocess_data(data):
    # Preprocess your data here
    texts = data['review'].values
    labels = data['sentiment'].values

    # Tokenize texts
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=100)

    # Encode labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    return padded_sequences, encoded_labels, tokenizer, tokenizer.word_index
