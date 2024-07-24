import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from FEDPAD_models import create_lstm_model, CentralServer, Client
from FEDPAD_utils import evaluate_model, setup_gpu
from Network_data import load_and_preprocess_data

# Set up GPU
setup_gpu()

# Parameters
sequence_length = 50
num_clients = 20
num_epochs = 20
num_rounds = 20

# Load and preprocess data
train_file = './data/Train.txt'
test_file = './data/Test.txt'

X_train, y_train = load_and_preprocess_data(train_file, sequence_length)
X_test, y_test = load_and_preprocess_data(test_file, sequence_length)

# Convert labels to binary format (normal vs anomaly)
y_train_binary = np.where(y_train == 'normal', 0, 1)
y_test_binary = np.where(y_test == 'normal', 0, 1)

# Number of features
num_features = X_train.shape[2]

# Train simple LSTM model
simple_model = create_lstm_model(sequence_length, num_features)
simple_model.fit(X_train, y_train_binary, epochs=num_epochs, batch_size=32, validation_split=0.1, verbose=1)
print("fin de l'entrainement du lstm simple")

# Split training data among clients
client_data = np.array_split(X_train, num_clients)
client_labels = np.array_split(y_train_binary, num_clients)

print('Creation du server central pour l apprentissage fédéré')
central_server = CentralServer(
    create_lstm_model,
    sequence_length,
    num_features,
    model_path='federated_model.h5'
)
print(f"Le modèle du serveur central est-il déjà entraîné ? {central_server.trained}")
clients = [Client(i, data, labels) for i, (data, labels) in enumerate(zip(client_data, client_labels))]
print(f"Nombre de clients créés : {len(clients)}")
print("Serveur central initialisé.")

print("Début de l'apprentissage fédéré")
# Federated learning
# if not central_server.trained:
for round in range(num_rounds):
    print(f"\n===== Round {round + 1} =====")
    
    print("Distributing model to clients...")
    model_weights = central_server.distribute_model()
    for client in clients:
        client.receive_model(model_weights, sequence_length, num_features)
    
    print("Training on each client...")
    for i, client in enumerate(clients):
        print(f"  Training on Client {i+1}...")
        client.train(epochs=num_epochs)
    
    print("- Collecting weights from clients...")
    weights_list = [client.get_weights() for client in clients]
    
    print("- Aggregating weights at the central server...")
    aggregated_weights = central_server.aggregate_weights(weights_list)
    
    print("- Updating global model...")
    central_server.update_model(aggregated_weights)
    
    # Save model after each round
    central_server.save_model()

    print("Federated learning completed. Final model saved.")
# else:
#     print("Pre-trained model loaded. No additional training needed.")

# Evaluate models
simple_model_params = {
    "Sequence length": sequence_length,
    "Epochs": num_epochs,
    "Number of features": num_features
}

simple_report, simple_plot = evaluate_model(simple_model, X_test, y_test_binary, simple_model_params, "Simple LSTM")

central_server_params = {
    "Sequence length": sequence_length,
    "Number of rounds": num_rounds,
    "Number of clients": num_clients,
    "Number of features": num_features
}

federated_report, federated_plot = evaluate_model(central_server.model, X_test, y_test_binary, central_server_params, "Central Server (FEDPAD)")

# Save figures
simple_report.savefig('Report_Simple_LSTM.png', dpi=300, bbox_inches='tight')
simple_plot.savefig('Predictions_Simple_LSTM.png', dpi=300, bbox_inches='tight')
federated_report.savefig('Report_FEDPAD_Framework.png', dpi=300, bbox_inches='tight')
federated_plot.savefig('Predictions_FEDPAD_Framework.png', dpi=300, bbox_inches='tight')

# Close all figures to free memory
plt.close('all')