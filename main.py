import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, LayerNorm, BatchNorm1d, LeakyReLU
from torch.utils.data import random_split

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import MessagePassing, global_mean_pool, GATConv as PyG_GATConv, TransformerConv

from sklearn.metrics import f1_score, classification_report

from source.loadData import GraphDataset
from source.utils import set_seed
from source.models import uWuModel
from source.losses import NoisyCrossEntropyLoss

set_seed()

def add_zeros(data):
    if data.x is None:
        num_nodes = data.edge_index.max().item() + 1 if data.edge_index.numel() > 0 else 0
        if num_nodes == 0 and hasattr(data, 'num_nodes') and data.num_nodes > 0:
            num_nodes = data.num_nodes
        elif num_nodes == 0:
             num_nodes = 1
        data.x = torch.zeros((num_nodes, 1), dtype=torch.float)
    return data

def train(train_loader, model, optimizer, criterion, device, save_checkpoints=False, checkpoint_path=None, current_epoch=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds, all_targets = [], []
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)

        # collect predictions and targets for F1 score
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(batch.y.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    if save_checkpoints and checkpoint_path and current_epoch is not None:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")
    
    return avg_loss, accuracy, f1

def evaluate(val_loader, model, device, calculate_accuracy=True):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            
            total_loss += loss.item()
            
            if calculate_accuracy:
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if calculate_accuracy else 0.0
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def inference(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions

def save_predictions(predictions, test_path):
    script_dir = os.getcwd() 
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def plot_training_progress(losses, accuracies, f1_scores, save_path, prefix=''):
    os.makedirs(save_path, exist_ok=True)
    
    # Plot losses, accuracies, and F1 scores
    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title(f'{prefix.capitalize()} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(accuracies)
    plt.title(f'{prefix.capitalize()} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(f1_scores)
    plt.title(f'{prefix.capitalize()} F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{prefix}_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()



def main(args):
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 5
    
    # Model init
    model = uWuModel(emb_dim=args.emb_dim, edge_input_dim=7, num_classes=6, dropout=args.drop_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = NoisyCrossEntropyLoss(args.noise_prob)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well

    # Define checkpoint path relative to the script's directory
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Load pre-trained model for inference
    if os.path.exists(checkpoint_path) and not args.train_path:
        #model.load_state_dict(torch.load(checkpoint_path))
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded best model from {checkpoint_path}")

       # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # If train_path is provided, train the model
    if args.train_path:
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size

        generator = torch.Generator().manual_seed(12)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        num_epochs = args.epochs
        best_val_accuracy, best_val_f1 = 0.0, 0.0
        train_losses, train_accuracies, train_f1_scores = [], [], []
        val_losses, val_accuracies, val_f1_scores = [], [], []

        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        for epoch in range(num_epochs):
            train_loss, train_acc, train_f1 = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )

            val_loss, val_acc, val_f1 = evaluate(val_loader, model, device, calculate_accuracy=True)

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            train_f1_scores.append(train_f1)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_f1_scores.append(val_f1)

            scheduler.step(val_f1)

            # Save best model based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path} (F1: {val_f1:.4f})")
                logging.info(f"Best model updated and saved at {checkpoint_path} (Val F1: {val_f1:.4f})")

        # plot training and validation progress
        plot_training_progress(train_losses, train_accuracies, train_f1_scores, os.path.join(logs_folder, "plots"), 'Training')
        plot_training_progress(val_losses, val_accuracies, val_f1_scores, os.path.join(logs_folder, "plotsVal"), 'Validation')

        # print final classification report
        print("\nFinal Validation Classification Report:")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
        report = classification_report(all_targets, all_preds, digits=4)
        print(report)

    # Generate predictions for the test set using the best model
    #model.load_state_dict(torch.load(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    predictions = inference(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 1)')
    parser.add_argument('--drop_ratio', type=float, default=0.2, help='dropout ratio (default: 0.2)')
    parser.add_argument('--emb_dim', type=int, default=128, help='dimensionality of hidden units in GNNs (default: 128)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--noise_prob', type=float, default=0.2, help='cross entropy loss noise (default: 0.2)')
    
    args = parser.parse_args()
    main(args)
