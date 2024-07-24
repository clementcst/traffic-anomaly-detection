import time
import psutil
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix,precision_recall_fscore_support
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns



def measure_performance(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_mem = process.memory_info().rss
        
        execution_time = end_time - start_time
        memory_used = end_mem - start_mem
        
        print(f"Temps d'exécution : {execution_time:.2f} secondes")
        print(f"Mémoire utilisée : {memory_used / (1024 * 1024):.2f} MB")
        
        return result
    return wrapper



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def measure_inference_time(model, test_loader):
#     model.eval()
#     start_time = time.time()
#     with torch.no_grad():
#         for batch_x, _ in test_loader:
#             _ = model(batch_x)
#     end_time = time.time()
#     return end_time - start_time

def measure_inference_time(model, test_loader, device):
    model.to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            _ = model(batch_x)
    end_time = time.time()
    return end_time - start_time


# def evaluate_model(model, test_loader, model_params, model_name, inference_time, num_parameters):
#     print(model_name+" evaluation in progress.. ")
#     model.eval()
#     predictions = []
#     all_predictions = []
#     all_true_labels = []

#     with torch.no_grad():
#         for batch_x, batch_y in test_loader:
#             outputs = model(batch_x)
#             _, predicted = torch.max(outputs, 1)
#             all_predictions.extend(predicted.cpu().numpy())
#             all_true_labels.extend(batch_y.cpu().numpy())
#             predictions.extend(outputs.cpu().numpy())

#     all_predictions = np.array(all_predictions)
#     all_true_labels = np.array(all_true_labels)
#     predictions = np.array(predictions).flatten()


#     accuracy = accuracy_score(all_true_labels, all_predictions)
#     cm = confusion_matrix(all_true_labels, all_predictions)
#     precision, recall, f1_score, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='binary')

#     report_fig = create_professional_report(model_params, accuracy, precision, recall, f1_score, cm, model_name, inference_time, num_parameters)
#     prediction_fig = create_prediction_plot(predictions, model_name)
    
#     return report_fig, prediction_fig

def evaluate_model(model, test_loader, model_params, model_name, inference_time, num_parameters):
    print(model_name+" evaluation in progress.. ")
    model.eval()
    predictions = []
    all_predictions = []
    all_true_labels = []

    # Determine the device of the model
    device = next(model.parameters()).device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Move input data to the same device as the model
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(batch_y.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    predictions = np.array(predictions).flatten()

    accuracy = accuracy_score(all_true_labels, all_predictions)
    cm = confusion_matrix(all_true_labels, all_predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='binary')

    report_fig = create_professional_report(model_params, accuracy, precision, recall, f1_score, cm, model_name, inference_time, num_parameters)
    prediction_fig = create_prediction_plot(predictions, model_name)
    
    return report_fig, prediction_fig


def create_professional_report(model_params, accuracy, precision, recall, f1_score, cm, model_name, inference_time, num_parameters):
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(20, 15), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Model Evaluation Report for {model_name} - {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}", fontsize=16 )

    # Performance metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    values = [accuracy, precision, recall, f1_score]
    
    ax1[0].bar(metrics, values)
    ax1[0].set_ylim(0, 1)
    ax1[0].set_title("Performance Metrics")
    ax1[0].set_ylabel("Score")
    for i, v in enumerate(values):
        ax1[0].text(i, v + 0.01, f'{v:.3f}', ha='center')

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1[1])
    ax1[1].set_title("Confusion Matrix")
    ax1[1].set_xlabel("Predicted")
    ax1[1].set_ylabel("Actual")
    ax1[1].set_xticklabels(['Normal', 'Anomaly'])
    ax1[1].set_yticklabels(['Normal', 'Anomaly'])

    # Model parameters table
    ax2[0].axis('off')
    ax2[1].axis('off')
    table_data = [
        ["Parameter", "Value"],
        ["Inference Time", f"{inference_time:.4f} s"],
        ["Number of Parameters", f"{num_parameters:,}"]
    ]
    table_data.extend([k, str(v)] for k, v in model_params.items())
    table = ax2[0].table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2[0].set_title("Model Information", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    return fig

def create_prediction_plot(predictions, model_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'Predictions - {model_name}', fontsize=16)

    # Échantillonnage des points (1 sur 100)
    sample_rate = 100
    sampled_predictions = predictions[::sample_rate]
    sampled_indices = np.arange(0, len(predictions), sample_rate)

    # Graphique de points échantillonnés
    ax1.scatter(sampled_indices, sampled_predictions, alpha=0.5, s=10)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    ax1.set_ylabel('Prediction Score')
    ax1.set_title('Sampled Predictions')
    ax1.legend()
    ax1.grid(True)

    # Graphique de densité
    sns.kdeplot(predictions, fill=True, ax=ax2)
    ax2.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
    ax2.set_xlabel('Prediction Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Prediction Distribution')
    ax2.legend()

    plt.tight_layout()
    return fig