import torch
import json
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import compute_metrics, get_device, get_project_paths
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
    print(f"Evaluation Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    # Get absolute path to this script's directory
    BASE_DIR, PROJECT_ROOT = get_project_paths()

    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    
    architectures = ['RNN', 'LSTM', 'Bidirectional LSTM']

    for arch in architectures:
        model_name = arch.replace(" ", "_") # handles spaces in model name
        with open(os.path.join(results_dir, f'{model_name}_best_model_config.json'), 'r') as f:
            config = json.load(f)

        sequence_length = config.get('sequence_length', 50)  # default to 50 if not found

        x_test_filename = f'test_pad_{sequence_length}.npy'

        X_test = np.load(os.path.join(output_dir, x_test_filename))
        y_test = np.load(os.path.join(output_dir, 'test_labels.npy'))

        model = load_model(get_model, config, os.path.join(results_dir, f'best_model_{model_name}.pt'))

        print(f"{arch} Model Performance")
        print(config)
        evaluate_model(model, X_test, y_test)
        print("\n")