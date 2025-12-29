from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer
from evaluate import load
import numpy as np
import os
from sklearn.model_selection import train_test_split
import optuna
import argparse
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from src.utils import device, StanceDataset, WeightedTrainer, load_model, evaluate_model, compute_metrics, predict_stance
from transformers import EarlyStoppingCallback


print(f"Using device: {device}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run article and summary matching.")
    parser.add_argument("--data", type=str, required=True, help="Dataset path (e.g., ./Data/datasets/Hebrew_stance_dataset_combined.csv).")
    parser.add_argument("--model", type=str, default="dicta-il/dictabert-sentiment", help="Model name for finetuning.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the model.")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials for Optuna hyperparameter search.")
    parser.add_argument("--method", type=str, default="balanced", help="Method for training (e.g., weighted, balanced).")
    args = parser.parse_args()

    try:
        # path_to_csv = './Data/Hebrew_stance_dataset_combined.csv'
        path_to_csv = args.data

        if not os.path.exists(path_to_csv):
            raise FileNotFoundError(f"Dataset file not found: {path_to_csv}")
            
        df = pd.read_csv(path_to_csv)
        print(f"Dataset loaded successfully. Total samples: {len(df)}")

        texts = df["Text"].tolist()
        labels = df["Topic"].tolist()  

        # Create dynamic label mapping
        unique_labels = sorted(set(labels))
        LABEL2ID = {label: idx for idx, label in enumerate(unique_labels)}
        ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
        print(f"Label mapping: {LABEL2ID}")

        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42
        )
        eval_texts, test_texts, eval_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )

        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(eval_texts)}")
        print(f"Test samples: {len(test_texts)}")

        # model_name = 'dicta-il/dictabert-sentiment'
        model_name = args.model
        _, tokenizer = load_model(model_name)

        train_dataset = StanceDataset(train_texts, train_labels, tokenizer, LABEL2ID)
        eval_dataset = StanceDataset(eval_texts, eval_labels, tokenizer, LABEL2ID)
        test_dataset = StanceDataset(test_texts, test_labels, tokenizer, LABEL2ID)

        # output_dir = 'fine_tuned_dictabert_topic_stance_balanced'
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if args.method == "weighted":
            # choose the rarest labels
            label_counts = Counter(train_dataset.labels)
            sorted_labels_by_freq = sorted(label_counts.items(), key=lambda x: x[1])
            rare_labels = [label for label, count in sorted_labels_by_freq[:33]]  # the 11 smallest

            print("Rare labels:", rare_labels)

            weights_array = np.ones(len(LABEL2ID))  
            # Compute weights for all classes
            all_classes = np.arange(len(LABEL2ID))
            class_weight_values = compute_class_weight(
                class_weight='balanced',
                classes=all_classes,
                y=np.array(train_dataset.labels)
            )

            # Initialize full weight array
            weights_array = np.ones(len(LABEL2ID))
            class_weight_dict = dict(zip(all_classes, class_weight_values))

            # Only update weights for rare classes
            for label_id in rare_labels:
                weights_array[label_id] = class_weight_dict[label_id]

            # Convert to tensor
            class_weights = torch.tensor(weights_array, dtype=torch.float).to(device)
# ------------------------------------------------------------ OpTuna  ------------------------------------------------------------
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(LABEL2ID),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True 
            )


        def hp_space_optuna(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
                "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
            }

        base_training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=10,
            eval_steps=50,
            save_steps=50,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            label_smoothing_factor=0.1,
        )

        print("\nStarting Optuna hyperparameter search...")
        if args.method == "weighted":
            trainer = WeightedTrainer(
                model_init=model_init,
                args=base_training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                processing_class=tokenizer,
                class_weights=class_weights,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
        else:
            trainer = Trainer(
                model_init=model_init,
                args=base_training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                processing_class=tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            hp_space=hp_space_optuna,
            backend="optuna",
            # n_trials=10
            n_trials=args.num_trials 
        )

        print("Best hyperparameters found:", best_trial.hyperparameters)
        print(f"Best trial score: {best_trial.objective}")

# ------------------------------------------------------------ Training ------------------------------------------------------------
        # Final training
        final_training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=10,
            eval_steps=50,
            save_steps=50,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            label_smoothing_factor=0.1,
            **best_trial.hyperparameters
        )
        if args.method == "weighted":
            final_trainer = WeightedTrainer(
                model_init=model_init,
                args=final_training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                processing_class=tokenizer,
                class_weights=class_weights,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
        else:
            final_trainer = Trainer(
                model_init=model_init,
                args=final_training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                processing_class=tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

        print("\nTraining with best hyperparameters...")
        final_trainer.train()
        final_trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

# ------------------------------------------------------------ Evaluation ------------------------------------------------------------
        # # Evaluation
        # print("\nEvaluating on test set...")
        # final_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
        # test_dataloader = DataLoader(test_dataset, batch_size=8)
        # test_results = evaluate_model(final_model, test_dataloader)

        # print(f"Test Accuracy: {test_results['accuracy']:.4f}")

        # # Single example prediction
        # sample_idx = 500
        # sample_text = test_texts[sample_idx]
        # sample_label = test_labels[sample_idx]
        # prediction = predict_stance(sample_text, final_model, tokenizer, ID2LABEL)

        # print("\nSample prediction:")
        # print(f"Text: {sample_text}")
        # print(f"True label: {sample_label}")
        # print(f"Predicted label: {prediction}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
