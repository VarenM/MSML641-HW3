import torch
import json
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import compute_metrics, get_device
from models import get_model

def load_model(model_class, config, model_path):
    device = get_device()
    model = model_class(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_data, test_labels, batch_size=32):
    device = next(model.parameters()).device
    dataset = TensorDataset(torch.tensor(test_data, dtype=torch.long), torch.tensor(test_labels, dtype=torch.float))
    loader = DataLoader(dataset, batch_size=batch_size)
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())
    metrics = compute_metrics(all_targets, all_preds)
    print("Evaluation Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    # Get absolute path to this script's directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Move up one directory to project root
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')

    X_test = np.load(os.path.join(output_dir, 'test_pad_50.npy'))
    y_test = np.load(os.path.join(output_dir, 'test_labels.npy'))

    # best config
    # config = {
    #     'vocab_size': 10000,
    #     'embed_size': 100,
    #     'hidden_size': 64,
    #     'num_layers': 2,
    #     'dropout': 0.5,
    #     'cell_type': 'Bidirectional LSTM',
    #     'bidirectional': True,
    #     'activation': 'sigmoid',
    #     'batch_size': 32,
    #     'epochs': 10,
    #     'optimizer': 'Adam',
    #     'lr': 0.001,
    #     'grad_clip': None   # since GradClip=None in this run
    # }

    results_dir = os.path.join(PROJECT_ROOT, 'results')

    with open(os.path.join(results_dir, 'best_model_config.json'), 'r') as f:
        config = json.load(f)

    model = load_model(get_model, config, os.path.join(results_dir, 'best_model.pt'))
    evaluate_model(model, X_test, y_test)