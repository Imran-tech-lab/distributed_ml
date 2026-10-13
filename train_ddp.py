import os
import sys
import warnings

# 1. Suppress tqdm progress bars and PyTorch DDP synchronization warnings
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import inspect
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import evaluate
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    EarlyStoppingCallback,
    pipeline
)
from transformers.trainer_callback import PrinterCallback

# 2. Callback for Formatted Table and Cross-GPU Memory Metrics
class DistributedTableLoggerCallback(TrainerCallback):
    def __init__(self):
        self.header_printed = False
        self.latest_train_loss = None
        self.in_training = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.in_training = True

    def on_train_end(self, args, state, control, **kwargs):
        self.in_training = False
        if args.process_index == 0:
            print("=" * 94 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.latest_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Prevent test-set evaluation from appending rows after training ends
        if not self.in_training:
            return

        # Synchronize and gather GPU metrics from Rank 0 and Rank 1
        dev = torch.cuda.current_device()
        torch.cuda.synchronize(dev)
        peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        local_rank = dist.get_rank() if dist.is_initialized() else 0

        if dist.is_initialized():
            gathered_peaks = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_peaks, (local_rank, peak_mb))
        else:
            gathered_peaks = [(0, peak_mb)]

        if args.process_index == 0 and metrics:
            if not self.header_printed:
                print("\n" + "=" * 94)
                print(f"| {'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Macro F1':^10} | {'GPU 0 Peak':^12} | {'GPU 1 Peak':^12} |")
                print("=" * 94)
                self.header_printed = True

            epoch = int(round(state.epoch)) if state.epoch is not None else 0
            tr_loss_str = f"{self.latest_train_loss:.4f}" if self.latest_train_loss is not None else "N/A"
            val_loss = metrics.get("eval_loss", 0.0)
            val_acc = metrics.get("eval_accuracy", 0.0)
            val_f1 = metrics.get("eval_macro_f1", 0.0)

            gathered_peaks.sort(key=lambda x: x[0])
            g0_str = f"{gathered_peaks[0][1]:.0f} MB" if len(gathered_peaks) > 0 else "N/A"
            g1_str = f"{gathered_peaks[1][1]:.0f} MB" if len(gathered_peaks) > 1 else "N/A"

            acc_str = f"{val_acc * 100:.2f}%"
            row = (
                f"| {epoch:^7} | {tr_loss_str:^10} | {val_loss:^10.4f} | "
                f"{acc_str:^9} | {val_f1:^10.4f} | {g0_str:^12} | {g1_str:^12} |"
            )
            print(row)

# 3. Custom Trainer with Inverse Class Weighting
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weight_tensor = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fn = nn.CrossEntropyLoss()
            
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    # 4. Label Map & Dataset Setup
    id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    dataset = load_dataset("dair-ai/emotion")
    dataset = dataset.rename_column("label", "labels")

    model_checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 5. Inverse Class Weights to prevent bias toward dominant classes
    train_labels = np.array(dataset["train"]["labels"])
    computed_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(computed_weights, dtype=torch.float32)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6. Model Initialization with Dropout Regularization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        classifier_dropout=0.2,
        hidden_dropout_prob=0.2
    )

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        macro_f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        return {"accuracy": acc, "macro_f1": macro_f1}

    # 7. TrainingArguments Configuration
    output_dir = "./best_emotion_model"
    warmup_steps = 125

    training_args_dict = {
        "output_dir": output_dir,
        "save_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 64,
        "num_train_epochs": 5,
        "weight_decay": 0.05,
        "warmup_steps": warmup_steps,
        "fp16": True,
        "ddp_find_unused_parameters": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "logging_strategy": "epoch",
        "disable_tqdm": True,
        "report_to": "none"
    }

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        training_args_dict["eval_strategy"] = "epoch"
    else:
        training_args_dict["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_dict)

    # 8. Instantiate Trainer with processing_class
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,  # Replaces deprecated tokenizer=tokenizer
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            DistributedTableLoggerCallback(),
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )

    trainer.remove_callback(PrinterCallback)

    # 9. Execute Distributed Training
    trainer.train()

    # 10. Final Evaluation on Unseen Test Set & Inference Demo
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    
    if trainer.is_world_process_zero():
        print("=== FINAL TEST EVALUATION (UNSEEN TEST SET) ===")
        print(f"Test Accuracy : {test_results['eval_accuracy'] * 100:.2f}%")
        print(f"Test Macro F1 : {test_results['eval_macro_f1']:.4f}\n")

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        print("=" * 65)
        print("DEMO INFERENCE ON NEW UNSEEN TEXT")
        print("=" * 65)

        classifier = pipeline(
            "text-classification",
            model=output_dir,
            tokenizer=output_dir,
            device=0
        )

        sample_sentences = [
            "I am so proud of my sister for graduating with honors today!",
            "I feel completely crushed, empty, and hopeless after that conversation.",
            "I can't believe they broke their word again, it makes my blood boil!",
            "Hearing footsteps outside my door in the dark made my heart race with panic.",
            "Wait, you actually won first place? I never saw that coming at all!",
            "Holding your hand makes me realize how deeply I cherish you in my life."
        ]

        results = classifier(sample_sentences)
        for sentence, result in zip(sample_sentences, results):
            print(f"Input : \"{sentence}\"")
            print(f"Output: Emotion = {result['label'].upper():<10} | Confidence: {result['score'] * 100:.2f}%\n")

if __name__ == "__main__":
    main()
