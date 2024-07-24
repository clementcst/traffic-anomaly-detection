import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path, sequence_length):
    # Charger les données
    column_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
                    "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
                    "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
                    "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
                    "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
    
    df = pd.read_csv(file_path, header=None, names=column_names)
    
    # Encoder les variables catégorielles
    le = LabelEncoder()
    categorical_columns = ['protocol_type', 'service', 'flag']
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Séparer les caractéristiques et les étiquettes
    X = df.drop(['label', 'difficulty'], axis=1)
    y = df['label']
    
    # Normaliser toutes les caractéristiques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Créer des séquences
    sequences = []
    labels = []
    for i in range(0, len(X_scaled) - sequence_length + 1):
        sequence = X_scaled[i:i+sequence_length]
        sequences.append(sequence)
        label = y.iloc[i+sequence_length-1]
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

# # Utilisation de la fonction
# train_file = './data/Train.txt'
# test_file = './data/test.txt'
# sequence_length = 50  # Vous pouvez ajuster cette valeur

# X, y = load_and_preprocess_data(train_file, test_file, sequence_length)

# # Séparation en ensembles d'entraînement et de test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("Forme des données d'entraînement:", X_train.shape)
# print("Forme des étiquettes d'entraînement:", y_train.shape)
# print("Forme des données de test:", X_test.shape)
# print("Forme des étiquettes de test:", y_test.shape)
