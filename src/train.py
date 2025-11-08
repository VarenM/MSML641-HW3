import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
import itertools
import pandas as pd
import json
import os
# from preprocess import load_tokenizer
from models import get_model
from utils import set_seed, get_device, compute_metrics, plot_metrics


def train_model(model, train_data, train_labels, valid_data, valid_labels, config, best_valid_loss_so_far, worst_valid_loss_so_far):
    device = get_device()
    model.to(device)
    optimizer_class = getattr(optim, config['optimizer'])
    optimizer = optimizer_class(model.parameters(), lr=config['lr'])
    criterion = torch.nn.BCELoss()
    grad_clip = config.get('grad_clip', None)
    batch_size = config['batch_size']

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.long), torch.tensor(train_labels, dtype=torch.float))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(torch.tensor(valid_data, dtype=torch.long), torch.tensor(valid_labels, dtype=torch.float))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    best_valid_loss = best_valid_loss_so_far
    worst_valid_loss = worst_valid_loss_so_far
    best_metrics = None
    epoch_losses = []  # Add list to store training loss per epoch

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0

        # start timing epochs
        if torch.cuda.is_available():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        else:
            start_time = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()


        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            duration = start_time.elapsed_time(end_time) / 1000  # Convert ms to seconds
        else:
            end_time = time.time()
            duration = end_time - start_time

        epoch_losses.append(epoch_loss / len(train_loader))  # Average training loss this epoch

        # Validation evaluation
        model.eval()
        valid_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        valid_loss /= len(valid_loader)
        valid_metrics = compute_metrics(all_targets, all_preds)

        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.3f}, Valid Loss={valid_loss:.3f}, "
              f"Valid Acc={valid_metrics['accuracy']:.4f}, Valid F1={valid_metrics['f1']:.4f}, Time={duration:.2f}s")

        # Save best model if validation loss improves
        model_name = model.cell_type.replace(" ", "_")

        # Get absolute path to this script's directory
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Move up one directory to project root
        PROJECT_ROOT = os.path.dirname(BASE_DIR)

        results_dir = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(results_dir, exist_ok=True)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_metrics = valid_metrics
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pt'))
            # Save config as JSON for reproducibility
            with open(os.path.join(results_dir, 'best_model_config.json'), 'w') as f:
                json.dump(config, f)
            # Save training loss curve for the best model
            with open(os.path.join(results_dir, f'{model_name}_best_model_loss_curve.json'), 'w') as f:
                json.dump(epoch_losses, f)

        elif valid_loss > worst_valid_loss:
            worst_valid_loss = valid_loss
            # Save training loss curve for the worst model
            with open(os.path.join(results_dir, f'{model_name}_worst_model_loss_curve.json'), 'w') as f:
                json.dump(epoch_losses, f)

    # return model, best_metrics, best_valid_loss, worst_valid_loss
    return model, best_metrics or valid_metrics, best_valid_loss, worst_valid_loss

def main():
    set_seed(42)

    architectures = ['RNN', 'LSTM', 'Bidirectional LSTM']
    activations = ['sigmoid', 'relu', 'tanh']
    optimizers = ['Adam', 'SGD', 'RMSprop']
    sequence_lengths = [25, 50, 100]
    grad_clippings = [None, 1.0]

    # Get absolute path to this script's directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Move up one directory to project root
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')

    results = []

    # Track best/worst valid loss per model type globally
    best_valid_loss = {arch: float('inf') for arch in architectures}
    worst_valid_loss = {arch: -float('inf') for arch in architectures}

    for arch, act, opt, seq_len, grad_clip in itertools.product(architectures, activations, optimizers, sequence_lengths, grad_clippings):
        print(f"Training {arch}, Activation={act}, Optimizer={opt}, SeqLen={seq_len}, GradClip={grad_clip}")

        X_train = np.load(f'{output_dir}/train_pad_{seq_len}.npy')
        y_train = np.load(f'{output_dir}/train_labels.npy')
        X_valid = np.load(f'{output_dir}/valid_pad_{seq_len}.npy')
        y_valid = np.load(f'{output_dir}/valid_labels.npy')

        config = {
            'vocab_size': 10000,
            'embed_size': 100,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'cell_type': arch,
            'bidirectional': (arch == 'Bidirectional LSTM'),
            'activation': act,
            'batch_size': 32,
            'epochs': 1,
            'optimizer': opt,
            'lr': 0.001,
            'grad_clip': grad_clip
        }

        model = get_model(config)

        start_time = time.time()
        trained_model, best_val_metrics, best_valid_loss[arch], worst_valid_loss[arch] = train_model(
            model, X_train, y_train, X_valid, y_valid, config,
            best_valid_loss[arch], worst_valid_loss[arch]
        )

        total_time = time.time() - start_time
        epoch_time = total_time / config['epochs']

        # Use metrics returned from train_model; no redundant evaluation call here
        results.append({
            'Model': arch,
            'Activation': act.capitalize(),
            'Optimizer': opt,
            'Seq Length': seq_len,
            'Grad Clipping': 'Yes' if grad_clip else 'No',
            'Accuracy': best_val_metrics['accuracy'],
            'F1': best_val_metrics['f1'],
            'Epoch Time (s)': round(epoch_time, 2)
        })

    df = pd.DataFrame(results)
    df.to_csv('results/metrics.csv', index=False)
    print("Experiment sweep complete. Results saved to results/metrics.csv")

    print("Building Plots...")
    plot_metrics()

if __name__ == "__main__":
    main()