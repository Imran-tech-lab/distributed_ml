import os
import inspect
import torch
import torch.nn as nn
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

# 1. Custom Callback to Monitor Dual T4 VRAM Footprints
class GpuProfilingCallback(TrainerCallback):
    """Logs allocated, peak, and reserved VRAM per GPU at the end of each epoch."""
    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and args.process_index == 0:
            print("\n--- GPU Memory Metrics ---")
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
                alloc = torch.cuda.memory_allocated(i) / (1024 ** 2)
                peak = torch.cuda.max_memory_allocated(i) / (1024 ** 2)
                res = torch.cuda.memory_reserved(i) / (1024 ** 2)
                print(f"  GPU {i} ({torch.cuda.get_device_name(i)}) -> Allocated: {alloc:7.1f} MB | Peak: {peak:7.1f} MB | Reserved: {res:7.1f} MB")
            print("--------------------------\n")

# 2. Custom Trainer with Balanced Class Weighting
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        # Strip labels from model_inputs so the model doesn't compute redundant unweighted loss
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
    # 3. Label Mappings for dair-ai/emotion (6 Classes)
    id2label = {
        0: "sadness", 
        1: "joy", 
        2: "love", 
        3: "anger", 
        4: "fear", 
        5: "surprise"
    }
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    dataset = load_dataset("dair-ai/emotion")
    # Standardize label column to 'labels' for Hugging Face models
    dataset = dataset.rename_column("label", "labels")

    model_checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 4. Compute Inverse Class Weights to prevent majority-class collapse
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

    # 5. Model Initialization with Regularization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        classifier_dropout=0.2,
        hidden_dropout_prob=0.2
    )

    # 6. Evaluation Metrics (Accuracy and Macro F1)
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        macro_f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        return {"accuracy": acc, "macro_f1": macro_f1}

    # 7. Configure TrainingArguments with Safe Parameters
    output_dir = "./best_emotion_model"
    
    # 16,000 samples / (32 per GPU * 2 GPUs) = 250 steps/epoch.
    # For 5 epochs = 1,250 total steps. 10% warmup = 125 steps.
    warmup_steps = 125

    training_args_dict = {
        "output_dir": output_dir,
        "save_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 64,
        "num_train_epochs": 5,
        "weight_decay": 0.05,            # L2 penalty to stop weights from over-memorizing
        "warmup_steps": warmup_steps,     # Direct steps instead of warmup_ratio
        "fp16": True,                    # Tensor Core acceleration for T4
        "load_best_model_at_end": True,   # Automatically restore checkpoint with peak Macro F1
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "logging_strategy": "epoch",
        "disable_tqdm": False,
        "report_to": "none"
    }

    # Inspect whether current environment uses eval_strategy (>=4.41) or evaluation_strategy (<4.41)
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        training_args_dict["eval_strategy"] = "epoch"
    else:
        training_args_dict["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_dict)

    # 8. Instantiate Trainer
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            GpuProfilingCallback(),
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )

    # 9. Train Model
    trainer.train()

    # 10. Final Evaluation and Sample Inference (Run on Rank 0)
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    
    if trainer.is_world_process_zero():
        print("\n" + "=" * 65)
        print("FINAL TEST EVALUATION (UNSEEN TEST SET)")
        print("=" * 65)
        print(f"Test Accuracy : {test_results['eval_accuracy'] * 100:.2f}%")
        print(f"Test Macro F1 : {test_results['eval_macro_f1']:.4f}")
        print("=" * 65 + "\n")

        # Save the finalized model and tokenizer
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # 11. Live Pipeline Predictions
        print("=" * 65)
        print("LIVE INFERENCE EXAMPLES")
        print("=" * 65)

        classifier = pipeline(
            "text-classification",
            model=output_dir,
            tokenizer=output_dir,
            device=0
        )

        test_sentences = [
            "I am so proud of my sister for graduating with highest honors today!",
            "I feel completely crushed, empty, and hopeless after that phone call.",
            "I can't believe they broke their promise again, it makes my blood boil!",
            "Hearing unexpected footsteps outside my door in the dark made my heart race.",
            "Wait, you actually won the national championship? I never saw that coming!",
            "Holding your hand makes me realize how deeply I cherish you in my life."
        ]

        results = classifier(test_sentences)
        for sentence, result in zip(test_sentences, results):
            print(f"Input : \"{sentence}\"")
            print(f"Output: Emotion = {result['label'].upper():<10} | Confidence: {result['score'] * 100:.2f}%\n")

if __name__ == "__main__":
    main()
