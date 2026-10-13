%%writefile train_ddp.py
import os
import time
import torch
from torch.optim import AdamW
import evaluate
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)
from accelerate import Accelerator
from accelerate.utils import set_seed

def log_gpu_memory(accelerator, step_name=""):
    """Logs allocated, peak, and reserved VRAM per GPU rank."""
    local_rank = accelerator.local_process_index
    device = accelerator.device
    
    torch.cuda.synchronize(device)
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    max_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    
    print(
        f"[{step_name}] Rank {local_rank} ({torch.cuda.get_device_name(device)}) -> "
        f"Allocated: {alloc:.1f} MB | Peak: {max_alloc:.1f} MB | Reserved: {reserved:.1f} MB"
    )

def main():
    # 1. Initialize Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(42)

    accelerator.print(f"=== Starting Distributed Training on {accelerator.num_processes} GPUs ===")
    accelerator.print(f"Distributed Backend: {accelerator.state.distributed_type}")

    # 2. Dataset & Tokenizer (Yelp Review Full: 5 classes)
    model_checkpoint = "bert-base-uncased"
    num_labels = 5
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    raw_datasets = load_dataset("yelp_review_full")
    train_subset = raw_datasets["train"].shuffle(seed=42).select(range(8000))
    eval_subset = raw_datasets["test"].shuffle(seed=42).select(range(2000))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    with accelerator.main_process_first():
        tokenized_train = train_subset.map(tokenize_fn, batched=True, remove_columns=["text"])
        tokenized_eval = eval_subset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    batch_size = 32
    train_dataloader = DataLoader(
        tokenized_train, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_eval, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )

    # 3. Model, Optimizer, and Scheduler
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels
    )
    
    # Using PyTorch native AdamW
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    num_epochs = 3
    total_training_steps = num_epochs * (len(train_dataloader) // accelerator.num_processes)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_training_steps * 0.1), 
        num_training_steps=total_training_steps
    )

    # 4. Prepare distributed wrappers
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    # 5. Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            running_loss += loss.detach().float()

        epoch_duration = time.time() - epoch_start_time
        samples_processed = len(train_subset)
        throughput = samples_processed / epoch_duration

        accelerator.wait_for_everyone()
        log_gpu_memory(accelerator, step_name=f"End of Epoch {epoch + 1}")

        accelerator.print(
            f"Epoch {epoch + 1}/{num_epochs} Complete | "
            f"Time: {epoch_duration:.2f}s | "
            f"Throughput: {throughput:.2f} samples/sec | "
            f"Avg Loss: {(running_loss / len(train_dataloader)):.4f}"
        )

        # 6. Evaluation Loop with Distributed Gathering
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            
            predictions = outputs.logits.argmax(dim=-1)
            gathered_preds, gathered_labels = accelerator.gather_for_metrics(
                (predictions, batch["label"])
            )
            accuracy_metric.add_batch(predictions=gathered_preds, references=gathered_labels)
            f1_metric.add_batch(predictions=gathered_preds, references=gathered_labels)

        eval_acc = accuracy_metric.compute()["accuracy"]
        eval_f1 = f1_metric.compute(average="macro")["f1"]

        accelerator.print(
            f"--> Epoch {epoch + 1} Eval Metrics: "
            f"Global Accuracy = {eval_acc * 100:.2f}% | "
            f"Macro F1 = {eval_f1:.4f}\n"
        )

    accelerator.print("=== Distributed Training Finished Successfully ===")

if __name__ == "__main__":
    main()
