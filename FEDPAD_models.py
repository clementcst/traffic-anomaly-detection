import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def create_lstm_model(sequence_length, num_features):
    model = Sequential([
        LSTM(128, input_shape=(sequence_length, num_features), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

class CentralServer:
    def __init__(self, model_creator, sequence_length, num_features, model_path):
        self.model_path = model_path
        self.model_creator = model_creator
        self.sequence_length = sequence_length
        self.num_features = num_features

        if os.path.exists(self.model_path):
            print(f"Existing model found. Loading from {self.model_path}")
            try:
                self.model = load_model(self.model_path, compile=False)
                self.model.compile(optimizer='adam', loss='mse')
                self.trained = True
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Initializing a new model.")
                self.model = self.model_creator(self.sequence_length, self.num_features)
                self.trained = False
        else:
            print("No pre-existing model found. Initializing a new model.")
            self.model = self.model_creator(self.sequence_length, self.num_features)
            self.trained = False

    def distribute_model(self):
        return self.model.get_weights()

    def aggregate_weights(self, weights_list):
        avg_weights = [np.mean(weights, axis=0) for weights in zip(*weights_list)]
        return avg_weights

    def update_model(self, aggregated_weights):
        self.model.set_weights(aggregated_weights)

    def save_model(self):
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

class Client:
    def __init__(self, client_id, data, labels):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.model = None

    def receive_model(self, model_weights, sequence_length, num_features):
        self.model = create_lstm_model(sequence_length, num_features)
        self.model.set_weights(model_weights)

    def train(self, epochs=50):
        self.model.fit(self.data, self.labels, epochs=epochs, batch_size=32, verbose=0)

    def get_weights(self):
        return self.model.get_weights()