import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
import itertools
import pandas as pd
import json
import os
from models import get_model
from utils import set_seed, get_device, compute_metrics, plot_metrics, get_project_paths

def train_model(model, train_data, train_labels, valid_data, valid_labels, config):
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

    best_valid_loss = float('inf')
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
        
        # Track best validation loss internally
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_metrics = valid_metrics

    # Return full epoch losses, best validation loss and metrics
    return model, best_metrics, best_valid_loss, epoch_losses

def main():
    set_seed(42)

    architectures = ['RNN', 'LSTM', 'Bidirectional LSTM']
    activations = ['sigmoid', 'relu', 'tanh']
    optimizers = ['Adam', 'SGD', 'RMSprop']
    sequence_lengths = [25, 50, 100]
    grad_clippings = [None, 1.0]
    epochs = 10

    # Get absolute path to this script's directory
    BASE_DIR, PROJECT_ROOT = get_project_paths()

    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results = []

    # Track best/worst valid loss per model type globally
    best_valid_loss = {arch: float('inf') for arch in architectures}
    worst_valid_loss = {arch: -float('inf') for arch in architectures}

    best_losses_curves = {}
    worst_losses_curves = {}

    for arch, act, opt, seq_len, grad_clip in itertools.product(architectures, activations, optimizers, sequence_lengths, grad_clippings):
        print(f"Training {arch}, Activation={act}, Optimizer={opt}, SeqLen={seq_len}, GradClip={grad_clip}")

        x_train_filename = f'train_pad_{seq_len}.npy'
        x_valid_filename = f'valid_pad_{seq_len}.npy'

        X_train = np.load(os.path.join(output_dir, x_train_filename))
        y_train = np.load(os.path.join(output_dir, 'train_labels.npy'))
        X_valid = np.load(os.path.join(output_dir, x_valid_filename))
        y_valid = np.load(os.path.join(output_dir, 'valid_labels.npy'))

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
            'epochs': epochs,
            'optimizer': opt,
            'lr': 0.001,
            'grad_clip': grad_clip,
            'sequence_length': seq_len
        }

        model = get_model(config)

        start_time = time.time()
        trained_model, best_val_metrics, valid_loss, epoch_losses = train_model(
            model, X_train, y_train, X_valid, y_valid, config
        )

        duration = time.time() - start_time

        # Update best model weights and loss curve if improved
        model_name = arch.replace(" ", "_") # handles spaces in model name

        if valid_loss < best_valid_loss[arch]:
            best_valid_loss[arch] = valid_loss
            torch.save(trained_model.state_dict(), os.path.join(results_dir, f'best_model_{model_name}.pt'))
            with open(os.path.join(results_dir, f'{model_name}_best_model_config.json'), 'w') as f:
                json.dump(config, f)
            with open(os.path.join(results_dir, f'{model_name}_best_model_loss_curve.json'), 'w') as f:
                json.dump(epoch_losses, f)

            best_losses_curves[arch] = epoch_losses

        # Update worst model if applicable
        if valid_loss > worst_valid_loss[arch]:
            worst_valid_loss[arch] = valid_loss
            with open(os.path.join(results_dir, f'{model_name}_worst_model_loss_curve.json'), 'w') as f:
                json.dump(epoch_losses, f)

            worst_losses_curves[arch] = epoch_losses

        results.append({
            'Model': arch,
            'Activation': act.capitalize(),
            'Optimizer': opt,
            'Seq Length': seq_len,
            'Grad Clipping': 'Yes' if grad_clip else 'No',
            'Accuracy': best_val_metrics['accuracy'],
            'F1': best_val_metrics['f1'],
            'Epoch Time (s)': round(duration / epochs, 2)
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
    print("Experiment sweep complete. Results saved to results/metrics.csv")

    print("Building Plots...")
    plot_metrics(epochs)

if __name__ == "__main__":
    main()