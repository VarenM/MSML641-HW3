import numpy as np
import pandas as pd
import json
import os
import random
import torch
import platform
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Returns (BASE_DIR, PROJECT_ROOT) adjusted automatically for local machine vs Google Colab.
def get_project_paths():
    # Detect if running in Google Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        # Colab environment
        PROJECT_ROOT = "/content"
        BASE_DIR = os.path.join(PROJECT_ROOT, "src")
    else:
        # Local environment: determine dynamically based on this file's path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(BASE_DIR)

    return BASE_DIR, PROJECT_ROOT

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(targets, preds, threshold=0.5):
    preds_bin = [1 if p > threshold else 0 for p in preds]
    accuracy = accuracy_score(targets, preds_bin)
    f1 = f1_score(targets, preds_bin, average='macro')
    return {"accuracy": accuracy, "f1": f1}

def report_hardware():
    mem_bytes = None
    try:
        import psutil
        mem_bytes = psutil.virtual_memory().total
    except ImportError:
        print("Install psutil for RAM info")
    info = {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "ram_gb": mem_bytes / (1024 ** 3) if mem_bytes else "Unknown",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
    }
    print("Hardware Info:", info)
    return info

def plot_metrics(epochs_num=10):
    # Get absolute path to this script's directory
    BASE_DIR, PROJECT_ROOT = get_project_paths()

    results_dir = os.path.join(PROJECT_ROOT, 'results')

    # Load sweep summary CSV
    df = pd.read_csv(os.path.join(results_dir, 'metrics.csv'))

    # Separate by model type
    rnn_df = df[df['Model'] == 'RNN']
    lstm_df = df[df['Model'] == 'LSTM']
    bilstm_df = df[df['Model'] == 'Bidirectional LSTM']

    def plot_accuracy_f1_vs_seq_length(df, model_name):
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = df.groupby('Seq Length')
        seq_lengths = sorted(df['Seq Length'].unique())
        acc_means = grouped['Accuracy'].mean()
        acc_stds = grouped['Accuracy'].std()
        f1_means = grouped['F1'].mean()
        f1_stds = grouped['F1'].std()

        ax.errorbar(seq_lengths, acc_means, yerr=acc_stds, fmt='-o', label='Accuracy')
        ax.errorbar(seq_lengths, f1_means, yerr=f1_stds, fmt='-o', label='F1 Score')
        ax.set_title(f'{model_name} Accuracy and F1 vs Sequence Length')
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()

        filename = f'{model_name}_accuracy_f1_vs_seq_length.png'
        plt.savefig(os.path.join(results_dir, filename))

        plt.close()

    def plot_training_loss(best_loss_curve_path, worst_loss_curve_path, model_name):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Load loss curves
        with open(best_loss_curve_path, 'r') as f:
            best_losses = json.load(f)
        with open(worst_loss_curve_path, 'r') as f:
            worst_losses = json.load(f)

        # Use lengths of loss curves to create epochs x-axis
        epochs_best = list(range(1, len(best_losses) + 1))
        epochs_worst = list(range(1, len(worst_losses) + 1))

        ax.plot(epochs_best, best_losses, marker='o', label='Best Model')
        ax.plot(epochs_worst, worst_losses, marker='s', label='Worst Model')
        
        ax.set_title(f'{model_name} Training Loss vs Epochs (Best vs Worst)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.grid(True)
        ax.legend()

        filename = f'{model_name}_training_loss_best_worst.png'
        plt.savefig(os.path.join(results_dir, filename))

        plt.close()

    # Plot accuracy & F1 vs Seq Length for each model type
    plot_accuracy_f1_vs_seq_length(rnn_df, 'RNN')
    plot_accuracy_f1_vs_seq_length(lstm_df, 'LSTM')
    plot_accuracy_f1_vs_seq_length(bilstm_df, 'Bidirectional_LSTM')

    # Determine best and worst models for loss curve plotting (approximate by accuracy)
    def get_loss_curve_path(model_row, best_or_worst):
        model_name = model_row['Model'].replace(" ", "_")

        filename = f'{model_name}_{best_or_worst}_model_loss_curve.json'

        return os.path.join(results_dir, filename)

    for model_name, model_df in [('RNN', rnn_df), ('LSTM', lstm_df), ('Bidirectional_LSTM', bilstm_df)]:
        best_acc_idx = model_df['Accuracy'].idxmax()
        worst_acc_idx = model_df['Accuracy'].idxmin()
        best_model = model_df.loc[best_acc_idx]
        worst_model = model_df.loc[worst_acc_idx]

        # Paths for best and worst loss curves
        best_loss_path = get_loss_curve_path(best_model, "best")
        worst_loss_path = get_loss_curve_path(worst_model, "worst")

        plot_training_loss(best_loss_path, worst_loss_path, model_name)

    print("All plots saved to the results/ directory.")
    
if __name__ == "__main__":
    report_hardware()