import os
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

# 1. Custom Callback for GPU VRAM and Utilization Logging
class GpuProfilingCallback(TrainerCallback):
    """Logs allocated and peak VRAM per GPU rank at each epoch boundary."""
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

# 2. Weighted Loss Trainer to Prevent Imbalance-Driven Overfitting
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            # Transfer class weights to the active GPU
            weight_tensor = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fn = nn.CrossEntropyLoss()
            
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    # 3. Label Map & Dataset Setup
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
    model_checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 4. Compute Balanced Class Weights from Training Split
    train_labels = np.array(dataset["train"]["label"])
    computed_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(computed_weights, dtype=torch.float32)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Model Initialization with Dropout Regularization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        classifier_dropout=0.2,
        hidden_dropout_prob=0.2
    )

    # 6. Metric Definitions
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        macro_f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        return {"accuracy": acc, "macro_f1": macro_f1}

    # 7. Trainer Arguments
    output_dir = "./best_emotion_model"
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,   # 32 * 2 GPUs = Global batch size of 64
        per_device_eval_batch_size=64,
        num_train_epochs=6,
        weight_decay=0.05,                # L2 penalty against overparameterized weights
        warmup_ratio=0.1,
        fp16=True,                        # Mixed precision for dual T4s
        load_best_model_at_end=True,       # Restores peak checkpoint automatically
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy="epoch",
        disable_tqdm=False,               # Native Trainer progress table
        report_to="none"
    )

    # 8. Instantiate Weighted Trainer with Callbacks
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
            GpuProfilingCallback(),
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )

    # 9. Train Model
    trainer.train()

    # 10. Final Test Set Evaluation
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    if trainer.is_world_process_zero():
        print("\n=== Final Test Evaluation on Unseen Test Split ===")
        print(f"Test Accuracy: {test_results['eval_accuracy'] * 100:.2f}%")
        print(f"Test Macro F1: {test_results['eval_macro_f1']:.4f}\n")

        # Save optimal checkpoint and tokenizer
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # 11. Inference Pipeline on Sample Inputs
        print("=" * 60)
        print("DEMO INFERENCE WITH BEST LOADED MODEL")
        print("=" * 60)
        
        classifier = pipeline(
            "text-classification",
            model=output_dir,
            tokenizer=output_dir,
            device=0
        )

        sample_texts = [
            "I am so proud of my sister for graduating with honors today!",
            "I feel completely crushed, hollow, and hopeless after hearing the bad news.",
            "I can't believe how horribly they treated us, it makes my blood boil!",
            "Hearing strange footsteps outside my bedroom door filled me with panic.",
            "Wait, you actually got the scholarship? I never saw that coming at all!",
            "Holding your hand makes me realize how deeply I treasure you in my life."
        ]

        predictions = classifier(sample_texts)
        for text, pred in zip(sample_texts, predictions):
            print(f"Input : \"{text}\"")
            print(f"Output: Emotion = {pred['label'].upper():<10} (Confidence: {pred['score'] * 100:.2f}%)\n")

if __name__ == "__main__":
    main()
