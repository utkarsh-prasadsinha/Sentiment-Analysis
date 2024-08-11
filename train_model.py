#train_model.py

# Suppressing Warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from app.preprocess import fetch_data, preprocess_data
import pickle

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Fetch and preprocess data
print(f"{Colors.OKBLUE}Fetching data from MongoDB...{Colors.ENDC}")
data = fetch_data()
print(f"{Colors.OKGREEN}Data fetched: {len(data)} records{Colors.ENDC}")

print(f"{Colors.OKBLUE}Preprocessing data...{Colors.ENDC}")
X, y, tokenizer, word_index = preprocess_data(data)
print(f"{Colors.OKGREEN}Data preprocessed: {X.shape[0]} samples{Colors.ENDC}")

# Split data
print(f"{Colors.OKBLUE}Splitting data into training and test sets...{Colors.ENDC}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"{Colors.OKGREEN}Training set: {X_train.shape[0]} samples{Colors.ENDC}")
print(f"{Colors.OKGREEN}Test set: {X_test.shape[0]} samples{Colors.ENDC}")

# Define the model
print(f"{Colors.OKBLUE}Defining the model...{Colors.ENDC}")
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=100))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
print(f"{Colors.OKBLUE}Compiling the model...{Colors.ENDC}")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print(f"{Colors.OKBLUE}Training the model...{Colors.ENDC}")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

print(f"{Colors.OKGREEN}Model training complete.{Colors.ENDC}")
print(f"{Colors.OKCYAN}Training history: {history.history}{Colors.ENDC}")

# Save the model and tokenizer
print(f"{Colors.OKBLUE}Saving the model and tokenizer...{Colors.ENDC}")
model.save('app/sentiment_model.h5')  # Ensure the path is correct
with open('app/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"{Colors.OKGREEN}Model and tokenizer saved successfully.{Colors.ENDC}")
