from transformers import (
    AutoModel, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    AutoModelForSequenceClassification
)
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, f1_score
import os

import argparse
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from src.utils import device, StanceDataset, WeightedTrainer, load_model, evaluate_model, compute_metrics, predict_stance
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

import tempfile
import shutil

class DictaLMClassifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Set the classifier weight and bias to match the encoder's dtype
        encoder_dtype = next(encoder.parameters()).dtype
        self.classifier.weight.data = self.classifier.weight.data.to(encoder_dtype)
        if self.classifier.bias is not None:
            self.classifier.bias.data = self.classifier.bias.data.to(encoder_dtype)
    
    def to(self, device):
        """Override to method to ensure all components move to device"""
        super().to(device)
        self.encoder = self.encoder.to(device)
        self.classifier = self.classifier.to(device)
        return self

    def forward(self, input_ids, attention_mask=None, labels=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(encoder_outputs, 'last_hidden_state'):
            last_hidden_state = encoder_outputs.last_hidden_state
        # If encoder_outputs is a tuple, the first element is usually the last_hidden_state
        elif isinstance(encoder_outputs, tuple):
            last_hidden_state = encoder_outputs[0]
        else:
            # Fallback or error if the structure is unexpected
            raise TypeError(f"Unexpected encoder output type: {type(encoder_outputs)}")

        pooled_output = last_hidden_state[:, 0]  # Use the [CLS] token's representation

        # Ensure pooled_output is the same dtype as the classifier weights before linear layer
        pooled_output = pooled_output.to(self.classifier.weight.dtype)
        
        # Get logits from the classifier
        final_logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            logits_for_loss_computation = final_logits.float()
            loss_fct = nn.CrossEntropyLoss() 
            loss = loss_fct(logits_for_loss_computation, labels)
            
        return (loss, final_logits)


def objective(trial, model_name, num_labels, tokenizer, train_dataset, eval_dataset, label2id, id2label, weights_array=None):
    """Optuna objective function for hyperparameter optimization."""
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    # Reduce the batch size options, as this is a major memory consumer
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 4) # Reduced max epochs for faster trials
    # Also consider tuning gradient_accumulation_steps
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])


    # Create a fresh model for each trial
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # Create a temporary directory for this trial that will be cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        trial_output_dir = os.path.join(temp_dir, f"trial_{trial.number}")

        training_args = TrainingArguments(
            output_dir=trial_output_dir,
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="no",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            load_best_model_at_end=False,
            metric_for_best_model="f1_macro",
            push_to_hub=False,
            report_to="none",
            save_total_limit=0,
            dataloader_num_workers=0,
            
            # --- KEY ADDITIONS FOR MEMORY OPTIMIZATION ---
            fp16=True,  # Enable mixed-precision training
            gradient_accumulation_steps=gradient_accumulation_steps, 
        )

        # Custom trainer with class weights if provided
        if weights_array is not None:
            class_weights = torch.tensor(weights_array, dtype=torch.float).to(device)
            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                processing_class=tokenizer,
                class_weights=class_weights,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                processing_class=tokenizer,
            )

        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
        trainer.train()

        eval_results = trainer.evaluate()
        
        del model
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return eval_results["eval_f1"]


def preprocess(example, tokenizer, label2id):
    combined = f"{example['sentence']} [SEP] {example['topic']}"
    inputs = tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs["label"] = label2id[example["stance"]]
    return inputs


def split_data_with_topic_exclusion(data, exclude_topics=["חופש ביטוי"], random_state=42):
    """
    Split data ensuring that the specified topics are only in the test set.
    
    Args:
        data: List of tuples (sentence, topic, label)
        exclude_topics: List of topics to exclude from train/validation sets
        random_state: Random state for reproducibility
    
    Returns:
        train_data, eval_data, test_data
    """
    # Convert to list if single topic is passed as string
    if isinstance(exclude_topics, str):
        exclude_topics = [exclude_topics]
    
    # Separate data by topic
    excluded_topic_data = [item for item in data if item[1] in exclude_topics]
    other_topics_data = [item for item in data if item[1] not in exclude_topics]
    
    print(f"Total samples: {len(data)}")
    print(f"Samples with excluded topics {exclude_topics}: {len(excluded_topic_data)}")
    print(f"Samples with other topics: {len(other_topics_data)}")
    
    # Print breakdown by excluded topic
    for topic in exclude_topics:
        topic_count = len([item for item in data if item[1] == topic])
        print(f"  - '{topic}': {topic_count} samples")
    
    # Check if we have enough data for the excluded topics
    if len(excluded_topic_data) == 0:
        raise ValueError(f"No data found for topics {exclude_topics}")
    
    # Split other topics data into train and validation
    if len(other_topics_data) == 0:
        raise ValueError("No data available for training after excluding the specified topics")
    
    # Get labels for stratification (only for other topics)
    other_labels = [item[2] for item in other_topics_data]
    
    # Check if we have enough samples for stratified split
    label_counts = Counter(other_labels)
    min_count = min(label_counts.values())
    if min_count < 2:
        print("Warning: Some labels have very few samples. Using random split instead of stratified split.")
        train_data, eval_data = train_test_split(
            other_topics_data, 
            test_size=0.2, 
            random_state=random_state
        )
    else:
        # Split other topics: 80% train, 20% validation
        train_data, eval_data = train_test_split(
            other_topics_data, 
            test_size=0.2, 
            random_state=random_state, 
            stratify=other_labels
        )
    
    # All excluded topic data goes to test set
    test_data = excluded_topic_data
    
    print(f"Train set size: {len(train_data)} (topics: {set(item[1] for item in train_data)})")
    print(f"Validation set size: {len(eval_data)} (topics: {set(item[1] for item in eval_data)})")
    print(f"Test set size: {len(test_data)} (topics: {set(item[1] for item in test_data)})")
    
    return train_data, eval_data, test_data


if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")

    # python -m scripts.stance_detection_model --data ./Data/datasets/topic_stance_dataset.csv --output_dir ./models/stance_detection_model --exclude_topic "חופש ביטוי"
    parser = argparse.ArgumentParser(description="Fine-Tune dictalm2.0.")
    parser.add_argument("--data", type=str, required=True, help="Dataset path ('./Data/datasets/topic_stance_dataset.csv').")
    parser.add_argument("--model", type=str, default="dicta-il/dictabert-sentiment", help="Model name for finetuning.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the model.")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials for Optuna hyperparameter search.")
    parser.add_argument("--method", type=str, default="weighted", help="Method for training (e.g., weighted, balanced).")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes to give weight to.")
    parser.add_argument("--quick_test", action="store_true", help="Run with reduced settings for testing")
    parser.add_argument("--exclude_topic", type=str, default="חופש ביטוי", help="Topic(s) to exclude from training and validation sets. Use comma-separated list for multiple topics.")
    args = parser.parse_args()

    # Parse exclude_topic argument - can be single topic or comma-separated list
    if args.exclude_topic:
        exclude_topics = [topic.strip() for topic in args.exclude_topic.split(',')]
    else:
        exclude_topics = ["חופש ביטוי"]

    # ------------------------------------------------------------------------ 
    # Loading the model 
    # ------------------------------------------------------------------------
    try:
        model_name = args.model
        num_labels = 3  # e.g. בעד / נגד / ניטרלי

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Define label2id & id2label
        LABEL2ID = {"בעד": 0, "נגד": 1, "נייטרלי": 2}
        ID2LABEL = {v: k for k, v in LABEL2ID.items()}

        path_to_csv = args.data

        if not os.path.exists(path_to_csv):
            raise FileNotFoundError(f"Dataset file not found: {path_to_csv}")
            
        df = pd.read_csv(path_to_csv)
        print(f"Dataset loaded successfully. Total samples: {len(df)}")

        # Print topic distribution before filtering
        print("\nTopic distribution in full dataset:")
        print(df['topic'].value_counts())

        sentences = df["sentence"].tolist()
        topics = df["topic"].tolist()  
        labels = df["stance"].tolist()  # בעד / נגד / ניטרלי

        data = list(zip(sentences, topics, labels))

        # Split data with topic exclusion
        print(f"\nExcluding topics {exclude_topics} from training and validation sets...")
        train_data, eval_data, test_data = split_data_with_topic_exclusion(
            data, 
            exclude_topics=exclude_topics,
            random_state=42
        )

        train_sentences, train_topics, train_labels = zip(*train_data)
        eval_sentences, eval_topics, eval_labels = zip(*eval_data)
        test_sentences, test_topics, test_labels = zip(*test_data)

        # Print label distribution in each set
        print(f"\nLabel distribution in training set: {Counter(train_labels)}")
        print(f"Label distribution in validation set: {Counter(eval_labels)}")
        print(f"Label distribution in test set: {Counter(test_labels)}")

        # Create HuggingFace datasets
        train_dataset = Dataset.from_dict({"sentence": train_sentences, "topic": train_topics, "stance": train_labels})
        eval_dataset = Dataset.from_dict({"sentence": eval_sentences, "topic": eval_topics, "stance": eval_labels})
        test_dataset = Dataset.from_dict({"sentence": test_sentences, "topic": test_topics, "stance": test_labels})

        dataset_dict = DatasetDict({"train": train_dataset, "validation": eval_dataset, "test": test_dataset})

        # Apply preprocessing with correct scope for label2id
        preprocess_with_args = lambda example: preprocess(example, tokenizer, LABEL2ID)
        tokenized_datasets = dataset_dict.map(preprocess_with_args, batched=False)

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"]
        
        # output_dir = './models/stance_detection'
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save experiment info
        experiment_info = {
            "excluded_topics": exclude_topics,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "test_size": len(test_dataset),
            "train_topics": list(set(train_topics)),
            "eval_topics": list(set(eval_topics)),
            "test_topics": list(set(test_topics)),
            "train_label_distribution": dict(Counter(train_labels)),
            "eval_label_distribution": dict(Counter(eval_labels)),
            "test_label_distribution": dict(Counter(test_labels))
        }
        
        import json
        with open(os.path.join(output_dir, "experiment_info.json"), "w", encoding="utf-8") as f:
            json.dump(experiment_info, f, ensure_ascii=False, indent=2)

        if args.quick_test:
            print("Running in QUICK TEST mode")
            args.num_trials = 2
            # Use smaller subset of data for testing
            train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))
            eval_dataset = eval_dataset.select(range(min(200, len(eval_dataset))))
        
        # Setup for training method
        weights_array = None
        if args.method == "weighted":
            # Choose the rarest labels
            label_counts = Counter(train_dataset["label"])
            sorted_labels_by_freq = sorted(label_counts.items(), key=lambda x: x[1])
            rare_labels = [label for label, count in sorted_labels_by_freq[:args.num_classes]]  # the smallest

            print("Rare labels:", rare_labels)

            # Compute weights for all classes
            all_classes = np.arange(len(LABEL2ID))
            class_weight_values = compute_class_weight(
                class_weight='balanced',
                classes=all_classes,
                y=np.array(train_dataset["label"])
            )

            # Initialize full weight array
            weights_array = np.ones(len(LABEL2ID))
            class_weight_dict = dict(zip(all_classes, class_weight_values))

            # Only update weights for rare classes
            for label_id in rare_labels:
                weights_array[label_id] = class_weight_dict[label_id]

            print(f"Class weights: {weights_array}")

        # ------------------------------------------------------------------------ 
        # OpTuna hyperparameter search 
        # ------------------------------------------------------------------------
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        )

        objective_fn = lambda trial: objective(
            trial, 
            model_name=model_name,
            num_labels=num_labels,
            tokenizer=tokenizer, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            label2id=LABEL2ID, 
            id2label=ID2LABEL,
            weights_array=weights_array
        )

        try:
            study.optimize(objective_fn, n_trials=args.num_trials)  
        except Exception as e:
            print(f"Optimization stopped due to: {e}")
            if len(study.trials) == 0:
                print("No successful trials completed. Using default hyperparameters.")
                # Set default hyperparameters
                best_params = {
                    "learning_rate": 2e-5,
                    "batch_size": 16,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.1,
                    "num_train_epochs": 5
                }
            else:
                best_params = study.best_params
        
        # Get best hyperparameters
        if len(study.trials) > 0:
            best_params = study.best_params
            print(f"Best hyperparameters from {len(study.trials)} trials: {best_params}")
        else:
            # Fallback default parameters
            best_params = {
                "learning_rate": 2e-5,
                "batch_size": 16,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "num_train_epochs": 5
            }
            print("Using default hyperparameters due to failed trials")


        # ------------------------------------------------------------------------ 
        # Training with best hyperparameters
        # ------------------------------------------------------------------------
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",  # Now we can save for the final model
            learning_rate=best_params["learning_rate"],
            per_device_train_batch_size=best_params["batch_size"],
            per_device_eval_batch_size=best_params["batch_size"],
            num_train_epochs=best_params["num_train_epochs"],  
            weight_decay=best_params["weight_decay"],
            warmup_ratio=best_params["warmup_ratio"],
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            push_to_hub=False,
            max_grad_norm=1.0,
            gradient_accumulation_steps=2,  # Fixed typo
            label_smoothing_factor=0.1,
            greater_is_better=True,
            save_total_limit=2,  # Keep only 2 best checkpoints
            dataloader_num_workers=0,
        )
            
        # Final model for training
        final_model =  AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True 
            )
        final_model.to(device) 

        # Training with best hyperparameters
        if weights_array is not None:
            class_weights = torch.tensor(weights_array, dtype=torch.float).to(device)
            trainer = WeightedTrainer(
                model=final_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                processing_class=tokenizer,
                class_weights=class_weights
            )
        else:
            trainer = Trainer(
                model=final_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                processing_class=tokenizer,
            )
            
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

        print("\nTraining with best hyperparameters...")
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

        # ------------------------------------------------------------------------ 
        # Evaluation
        # ------------------------------------------------------------------------

        test_results = trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")
        
        # Save test results
        with open(os.path.join(output_dir, "model_test_results.txt"), "w") as f:
            f.write(f"Experiment: Excluded topics {exclude_topics} from training\n")
            f.write("="*50 + "\n")
            for key, value in test_results.items():
                f.write(f"{key}: {value}\n")
            f.write("\nDataset splits:\n")
            f.write(f"Train topics: {list(set(train_topics))}\n")
            f.write(f"Test topics: {list(set(test_topics))}\n")

        print(f"\nExperiment completed!")
        print(f"The model was trained without seeing topics {exclude_topics}")
        print(f"Test evaluation was performed only on topics {exclude_topics}")
        print(f"This tests the model's ability to generalize to unseen topics.")

    except Exception as e:
        print(f"Error during training: {e}")
        raise