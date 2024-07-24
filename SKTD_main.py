import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from SKTD_utils import measure_inference_time, count_parameters, evaluate_model
from SKTD_models import MSSTRNet, LENet, train_msstrnet, train_lenet_kd
from Network_data import load_and_preprocess_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Chargement et prétraitement des données
train_file = './data/Train.txt'
test_file = './data/Test.txt'
sequence_length = 50


X_train, y_train = load_and_preprocess_data(train_file, sequence_length)
X_test, y_test = load_and_preprocess_data(test_file, sequence_length)

# Conversion des étiquettes en format binaire (normal vs anomalie)
y_train_binary = np.where(y_train == 'normal', 0, 1)
y_test_binary = np.where(y_test == 'normal', 0, 1)

# Séparation de l'ensemble d'entraînement en entraînement et validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_binary, test_size=0.2, random_state=42)

# Conversion en tenseurs PyTorch
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test_binary)

# Création des datasets et dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4)

# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_dataset, batch_size=128, num_workers=0)
# test_loader = DataLoader(test_dataset, batch_size=128, num_workers=0)

# Définition des paramètres du modèle
input_size = X_train.shape[2]  # Nombre de caractéristiques
hidden_size = 64
num_classes = 2
num_blocks = 3
num_epochs = 100

# Avant l'entraînement du modèle MSSTRNet
msstrnet = MSSTRNet(input_size, hidden_size, num_classes, num_blocks).to(device)
msstrnet_path = 'msstrnet_model.pth'
if os.path.exists(msstrnet_path):
    msstrnet.load_state_dict(torch.load(msstrnet_path, map_location=device))
    print("Modèle MSSTRNet loaded.")
else:
    print("Aucun modèle MSSTRNet préexistant trouvé. Démarrage d'un nouvel entraînement.")
    print("Entraînement du modèle MSSTRNet (teacher):")
    msstrnet = train_msstrnet(msstrnet, train_loader, val_loader, epochs=num_epochs, lr=0.001, device=device)
    torch.save(msstrnet.state_dict(), 'msstrnet_model.pth')
    print("Modèle MSSTRNet saved.")

# Entraînement du modèle LENet (student) avec KD
lenet = LENet(input_size, hidden_size, num_classes).to(device)
lenet_path = 'lenet_model.pth'
if os.path.exists(lenet_path):
    lenet.load_state_dict(torch.load(lenet_path, map_location=device))
    print("Modèle LENet loaded.")
else:
    print("Aucun modèle LENet préexistant trouvé. Démarrage d'un nouvel entraînement.")
    print("Entraînement du modèle LENet (student) avec KD:")
    # lenet = train_lenet_kd(msstrnet, lenet, train_loader, val_loader, epochs=num_epochs, lr=0.001, temperature=5.0, lambda_kd=0.7)
    lenet = train_lenet_kd(msstrnet, lenet, train_loader, val_loader, epochs=num_epochs, lr=0.001, temperature=5.0, lambda_kd=0.7, device=device)
    torch.save(lenet.state_dict(), lenet_path)
    print("Modèle LENet saved.")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    msstrnet = nn.DataParallel(msstrnet)
    lenet = nn.DataParallel(lenet)

model_params = {
    "Sequence length": sequence_length,
    "Epochs": num_epochs,
    "Learning rate": 0.001,
    "Hidden size": hidden_size,
    "Number of blocks (MSSTRNet)": num_blocks
}

# Évaluation des modèles sur l'ensemble de test
inference_time_msstrnet = measure_inference_time(msstrnet, test_loader,device)
num_parameters_msstrnet = count_parameters(msstrnet)
msstrnet_report, msstrnet_plot = evaluate_model(msstrnet, test_loader, model_params, "MSSTRNet", inference_time_msstrnet, num_parameters_msstrnet)

inference_time_lenet = measure_inference_time(lenet, test_loader,device)
num_parameters_lenet = count_parameters(lenet)
lenet_report, lenet_plot = evaluate_model(lenet, test_loader, model_params, "LENet", inference_time_lenet, num_parameters_lenet)

# Sauvegarde des graphiques
msstrnet_report.savefig('Report_MSSTRNet_Model.png', dpi=300, bbox_inches='tight')
msstrnet_plot.savefig('Predictions_MSSTRNet_Model.png', dpi=300, bbox_inches='tight')
lenet_report.savefig('Report_LENet_Model.png', dpi=300, bbox_inches='tight')
lenet_plot.savefig('Predictions_LENet_Model.png', dpi=300, bbox_inches='tight')

# Fermer toutes les figures pour libérer la mémoire
plt.close('all')

