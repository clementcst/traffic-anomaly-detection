import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU found. Running on CPU.")

def evaluate_model(model, X_test, y_test, model_params, model_name):
    predictions = model.predict(X_test).flatten()
    all_predictions = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(y_test, all_predictions)
    cm = confusion_matrix(y_test, all_predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, all_predictions, average='binary')

    report_fig = create_professional_report(model_params, accuracy, precision, recall, f1_score, cm, model_name)
    prediction_fig = create_prediction_plot(predictions, model_name)

    return report_fig, prediction_fig

def create_professional_report(model_params, accuracy, precision, recall, f1_score, cm, model_name):
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(20, 15), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Model Evaluation Report for {model_name} - {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}", fontsize=16)

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
    table_data = [["Parameter", "Value"]]
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

    # Sample points (1 out of 100)
    sample_rate = 100
    sampled_predictions = predictions[::sample_rate]
    sampled_indices = np.arange(0, len(predictions), sample_rate)

    # Scatter plot of sampled points
    ax1.scatter(sampled_indices, sampled_predictions, alpha=0.5, s=10)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    ax1.set_ylabel('Prediction Score')
    ax1.set_title('Sampled Predictions')
    ax1.legend()
    ax1.grid(True)

    # Density plot
    sns.kdeplot(predictions, fill=True, ax=ax2)
    ax2.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
    ax2.set_xlabel('Prediction Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Prediction Distribution')
    ax2.legend()

    plt.tight_layout()
    return fig