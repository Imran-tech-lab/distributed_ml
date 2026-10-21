import os
import sys
import warnings

# Suppress nested tqdm serialization bars and PyTorch NCCL warnings
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

# 1. Custom Callback for ASCII Table Logging & Multi-GPU Memory Gathering
class DistributedTableLoggerCallback(TrainerCallback):
    def __init__(self):
        self.header_printed = False
        self.latest_train_loss = None
        self.in_training = False
        self.trainer = None  # Bound after trainer instantiation

    def on_train_begin(self, args, state, control, **kwargs):
        self.in_training = True

    def on_train_end(self, args, state, control, **kwargs):
        self.in_training = False
        if args.process_index == 0:
            print("=" * 110 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.latest_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.in_training:
            return

        # Synchronize and gather peak VRAM from GPU 0 and GPU 1
        dev = torch.cuda.current_device()
        torch.cuda.synchronize(dev)
        peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        local_rank = dist.get_rank() if dist.is_initialized() else 0

        if dist.is_initialized():
            gathered_peaks = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_peaks, (local_rank, peak_mb))
        else:
            gathered_peaks = [(0, peak_mb)]

        # Synchronize Train Accuracy counts across both ranks
        if self.trainer is not None:
            train_acc = self.trainer.get_and_reset_train_acc()
        else:
            train_acc = 0.0

        if args.process_index == 0 and metrics:
            if not self.header_printed:
                print("\n" + "=" * 110)
                print(f"| {'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Macro F1':^10} | {'GPU 0 Peak':^14} | {'GPU 1 Peak':^14} |")
                print("=" * 110)
                self.header_printed = True

            epoch = int(round(state.epoch)) if state.epoch is not None else 0
            
            # Retrieve latest training loss safely
            tr_loss_str = "N/A"
            if self.latest_train_loss is not None:
                tr_loss_str = f"{self.latest_train_loss:.4f}"
            else:
                for entry in reversed(state.log_history):
                    if "loss" in entry:
                        tr_loss_str = f"{entry['loss']:.4f}"
                        break

            val_loss = metrics.get("eval_loss", 0.0)
            val_acc = metrics.get("eval_accuracy", 0.0)
            val_f1 = metrics.get("eval_macro_f1", 0.0)

            gathered_peaks.sort(key=lambda x: x[0])
            g0_str = f"{gathered_peaks[0][1]:.0f} MB" if len(gathered_peaks) > 0 else "N/A"
            g1_str = f"{gathered_peaks[1][1]:.0f} MB" if len(gathered_peaks) > 1 else "N/A"

            tr_acc_str = f"{train_acc * 100:.2f}%"
            val_acc_str = f"{val_acc * 100:.2f}%"

            row = (
                f"| {epoch:^7} | {tr_loss_str:^10} | {tr_acc_str:^10} | {val_loss:^10.4f} | "
                f"{val_acc_str:^10} | {val_f1:^10.4f} | {g0_str:^14} | {g1_str:^14} |"
            )
            print(row)

# 2. Weighted Trainer with Train Accuracy Accumulation
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.train_correct = 0
        self.train_total = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")

        # Accumulate training accuracy on each rank during training mode
        if model.training and labels is not None:
            with torch.no_grad():
                preds = logits.detach().argmax(dim=-1)
                self.train_correct += (preds == labels).sum().item()
                self.train_total += labels.size(0)

        # Apply balanced class weights to prevent majority-class collapse
        if self.class_weights is not None:
            weight_tensor = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def get_and_reset_train_acc(self):
        """Synchronizes accumulated training counts across all DDP ranks."""
        dev = torch.cuda.current_device()
        local_counts = torch.tensor([self.train_correct, self.train_total], device=dev, dtype=torch.float32)
        
        if dist.is_initialized():
            gathered_counts = [torch.zeros_like(local_counts) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_counts, local_counts)
            tot_correct = sum(g[0].item() for g in gathered_counts)
            tot_samples = sum(g[1].item() for g in gathered_counts)
        else:
            tot_correct = self.train_correct
            tot_samples = self.train_total

        acc = (tot_correct / tot_samples) if tot_samples > 0 else 0.0
        
        # Reset counters for the upcoming epoch
        self.train_correct = 0
        self.train_total = 0
        return acc

def main():
    # 3. Label Map & Dataset Setup (6 Emotion Classes)
    id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    dataset = load_dataset("dair-ai/emotion")
    dataset = dataset.rename_column("label", "labels")

    model_checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 4. Inverse Class Weights to prevent bias toward dominant classes
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
    data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
)


    # 5. Model Initialization with Regularization
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

    # 6. TrainingArguments Configuration
    output_dir = "./best_emotion_model"
    warmup_steps = 125

    training_args_dict = {
    "output_dir": output_dir,
    "save_strategy": "epoch",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 64,  # was 32; applies to each GPU
    "per_device_eval_batch_size": 128,
    "num_train_epochs": 3,
    "weight_decay": 0.05,
    "warmup_ratio": 0.06,               # scales with the number of training steps
    "fp16": True,
    "ddp_find_unused_parameters": False,
    "load_best_model_at_end": True,
    "metric_for_best_model": "macro_f1",
    "greater_is_better": True,
    "save_total_limit": 1,
    "logging_strategy": "epoch",
    "disable_tqdm": True,
    "report_to": "none",
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "group_by_length": True,
}

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        training_args_dict["eval_strategy"] = "epoch"
    else:
        training_args_dict["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_dict)

    table_logger = DistributedTableLoggerCallback()

    # 7. Instantiate Trainer using processing_class
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            table_logger,
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )

    # Bind trainer reference to callback for cross-GPU metric extraction
    table_logger.trainer = trainer
    trainer.remove_callback(PrinterCallback)

    # 8. Run Distributed Training
    trainer.train()

    # 9. Final Test Set Evaluation & Demo Inference
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
