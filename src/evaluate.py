import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch 
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from train import encoded_val_data, tokenizer

def evaluate(model, dataloader, device):
    """Evaluate the model on the given dataloader.
    Args:
        model: The trained model to evaluate.
        dataloader: DataLoader for the evaluation dataset.
        device: Device to run the evaluation on (CPU or GPU).
        
        Returns: 
            A dictionary containing accuracy, precision, recall, F1-score, and AUC.
    """
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to the specified device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss
            loss_val_total += loss.item()

            # Move logits and labels to CPU and convert to numpy arrays
            logits = logits.detach().cpu().numpy()
            label_ids = batch['labels'].cpu().numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_vals.append(label_ids)

    # Calcualte the  loss average
    loss_val_avg = loss_val_total / len(dataloader)

    # Combine the prediction and labels from all batches
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    # Get the predicted class by taking the argmax of the logists
    preds_flat = np.argmax(predictions, axis=1).flatten()

    # Calculate metrics
    accuracy = accuracy_score(true_vals, preds_flat)
    precision = precision_score(true_vals, preds_flat)
    recall = recall_score(true_vals, preds_flat)
    f1 = f1_score(true_vals, preds_flat)

    print(f'Validation loss: {loss_val_avg}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score (weighted): {f1}')

    return loss_val_avg, accuracy, precision, recall, f1

# Load the trained model from the checkpoint
final_checkpoint = "../models/checkpoint-1491"
model = AutoModelForSequenceClassification.from_pretrained(final_checkpoint)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
model.to(device)

# Reomve the columns that are not needed for evaluation
encoded_val_data = encoded_val_data.remove_columns(['review', '__index_level_0__'])

# Create DataLoader for the validation dataset
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
val_dataloader = DataLoader(encoded_val_data, batch_size=16, collate_fn=data_collator)

# Call the evaluate function
evaluate(model, val_dataloader, device)