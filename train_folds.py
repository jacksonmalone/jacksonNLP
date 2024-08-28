import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
import os
import torch
from safetensors.torch import load_file, save_file
import random

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# this line below should be True, using the GPU
print(torch.cuda.is_available())
# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Load dataset from CSV
dataset = pd.read_csv('college_majors_dataset_updated.csv')
print("Data loaded!")

# Encode labels
label_encoder = LabelEncoder()
dataset['label'] = label_encoder.fit_transform(dataset['major'])
print("Labels encoded!")

X = dataset['text']  # Assuming 'text' is the column with the text data
y = dataset['label']  # Target labels

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
print("Tokenizer initialized!")

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)

# Define the directory where you want to save/load the model
saved_model_directory = "./saved_model_directory"

# Initialize EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)

# Load pre-trained model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_)).to(device)
print("Pre-trained model loaded!")
print("Model moved to device:", device)

# Cross-Validation
n_splits = 4  # Number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_results = []

user_input = input("Type 'train' if you would like to train the model, or 'pass' to pass training and classify texts: ")

if user_input.lower() == "train":
  for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/{n_splits}")
    
    # Split data into training and validation sets
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Reset the index of the training and validation sets
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    
    # Convert to Hugging Face Datasets
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    val_df = pd.DataFrame({'text': X_val, 'label': y_val})
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Set the format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold + 1}',
        eval_strategy='steps',
        max_steps=800,
        eval_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=40,
        weight_decay=0.01,
        logging_dir=f'./logs_fold_{fold + 1}',
        logging_steps=50,
        save_strategy='steps',
        save_steps=50,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        fp16=True,
        gradient_accumulation_steps=2,
        lr_scheduler_type='linear'
    )
    
    # Define compute_metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        probs = torch.nn.functional.softmax(torch.tensor(eval_pred.predictions), dim=-1).numpy()
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

        try:
            roc_auc = roc_auc_score(labels, probs, multi_class='ovr')
        except ValueError:
            roc_auc = np.nan  # Handle the case where ROC-AUC cannot be computed
        
        # Ensure target names match the number of classes
        target_names = label_encoder.classes_  # Adjust this to match your label encoding
        class_report = classification_report(labels, predictions, target_names=target_names, labels=np.arange(len(target_names)), output_dict=True, zero_division=0)
        
        # Extract relevant metrics from the classification report
        report_accuracy = class_report['accuracy']
        report_precision = class_report['weighted avg']['precision']
        report_recall = class_report['weighted avg']['recall']
        report_f1 = class_report['weighted avg']['f1-score']

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "report_accuracy": report_accuracy,
            "report_precision": report_precision,
            "report_recall": report_recall,
            "report_f1": report_f1
        }
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    print(f"Fold {fold + 1} Validation Accuracy: {results['eval_accuracy']}")
    fold_results.append(results)
    
    # Save the model for this fold
    os.makedirs(saved_model_directory, exist_ok=True)
    fold_safetensors_file = os.path.join(saved_model_directory, f"model_fold_{fold + 1}.safetensors")
    state_dict = model.state_dict()
    save_file(state_dict, fold_safetensors_file)
    print(f"Model for fold {fold + 1} saved to {fold_safetensors_file}!")
    
    # Re-initialize the model for the next fold
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_))


  # Extract the individual metrics for each fold
  accuracies = [result['eval_accuracy'] for result in fold_results]
  precisions = [result['eval_precision'] for result in fold_results]
  recalls = [result['eval_recall'] for result in fold_results]
  f1s = [result['eval_f1'] for result in fold_results]
  roc_aucs = [result.get('eval_roc_auc', np.nan) for result in fold_results]  # Handle possible NaN values

  
  # Compute the average metrics across folds
  avg_accuracy = np.mean(accuracies)
  avg_precision = np.mean(precisions)
  avg_recall = np.mean(recalls)
  avg_f1 = np.mean(f1s)
  avg_roc_auc = np.nanmean(roc_aucs)  # Use nanmean to ignore NaN values

  print(f"Average Validation Accuracy: {avg_accuracy}")
  print(f"Average Validation Precision: {avg_precision}")
  print(f"Average Validation Recall: {avg_recall}")
  print(f"Average Validation F1 Score: {avg_f1}")
  print(f"Average Validation ROC AUC Score: {avg_roc_auc}")

  # Find the best fold for each metric
  best_accuracy_fold = np.argmax([result['eval_accuracy'] for result in fold_results]) + 1
  best_precision_fold = np.argmax([result['eval_precision'] for result in fold_results]) + 1
  best_recall_fold = np.argmax([result['eval_recall'] for result in fold_results]) + 1
  best_f1_fold = np.argmax([result['eval_f1'] for result in fold_results]) + 1
  best_roc_auc_fold = np.argmax([result['eval_roc_auc'] for result in fold_results]) + 1

  print(f"Best Accuracy Fold: {best_accuracy_fold}")
  print(f"Best Precision Fold: {best_precision_fold}")
  print(f"Best Recall Fold: {best_recall_fold}")
  print(f"Best F1 Score Fold: {best_f1_fold}")
  print(f"Best ROC AUC Score Fold: {best_roc_auc_fold}")

elif user_input.lower() == "pass":
  pass
else:
  print("Please type either 'train' or 'pass'.")
  exit()

# Load the best-performing model
#best_fold = 4  # Change this to the desired metric fold number, or if training, comment this out and use one of the "best_[insert metric here]_fold"
best_model_path = os.path.join(saved_model_directory, f"model_fold_{best_accuracy_fold}.safetensors") # change model_fold to best_accuracy_fold, best_precision_fold, best_recall_fold, or best_f1_fold if training, change it to best_fold if passing training
best_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_)).to(device)
best_model.load_state_dict(load_file(best_model_path))
print(f"Best performing model from fold {best_accuracy_fold} loaded for inference!") # same as fold from best_model_path

# Perform text classification on new data
def classify_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = best_model(**inputs)
    # Get top 2 predictions
    top_k = torch.topk(outputs.logits, k=2, dim=-1)
    # Detach the tensors before converting to numpy
    top_indices = top_k.indices.detach().cpu().numpy()
    top_scores = top_k.values.detach().cpu().numpy()
    # Flatten the indices array for inverse_transform
    flat_indices = top_indices.flatten()
    # Decode the predictions to original labels
    flat_labels = label_encoder.inverse_transform(flat_indices)
    # Reshape the flat_labels back to the original shape
    top_labels = flat_labels.reshape(top_indices.shape)
    return top_labels, top_scores

# Example texts for classification
new_texts = ["This is where you can add texts for the model to classify."]
predicted_labels, predicted_scores = classify_texts(new_texts)

for i, text in enumerate(new_texts):
    print(f"Text: {text}")
    print(f"Top 2 Predicted Labels: {predicted_labels[i]}")
    print(f"Top 2 Predicted Scores: {predicted_scores[i]}")
