from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from data_loader import df


# Splitting the dataset into training and validation sets
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['sentiment'])

# Rename the sentiment column to labels for compatibility
train_df = train_df.rename(columns={'sentiment': 'labels'})
val_df = val_df.rename(columns={'sentiment': 'labels'})

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenization
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize(batch):
    return tokenizer(batch['review'], padding=True, truncation=True)

print(tokenize({'review': df['review'].tolist()}))

# Tokenizing the Training dataset
encoded_data_train = train_dataset.map(tokenize, batched=True, batch_size=None)

# Tokenizing the validation dataset
encoded_val_data = val_dataset.map(tokenize, batched=True, batch_size=None)

# Training with trainer API
if __name__ == "__main__":
    training_args = TrainingArguments(output_dir="./results", do_eval=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_data_train,
        eval_dataset=encoded_val_data
    )

    trainer.train()