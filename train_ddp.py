import os
import time
import torch
from torch.optim import AdamW
import evaluate
from tqdm.auto import tqdm
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
    """Collects VRAM usage across all ranks and prints cleanly from rank 0."""
    device = accelerator.device
    torch.cuda.synchronize(device)
    
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    max_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    
    # Gather metrics across GPU 0 and GPU 1
    local_metrics = torch.tensor([alloc, max_alloc, reserved], device=device, dtype=torch.float32)
    all_metrics = accelerator.gather(local_metrics).view(-1, 3).cpu().numpy()
    
    if accelerator.is_main_process:
        print(f"\n--- GPU Memory Profiling [{step_name}] ---")
        for rank_id, (a, m, r) in enumerate(all_metrics):
            print(
                f"  GPU {rank_id} (Rank {rank_id}) -> "
                f"Allocated: {a:7.1f} MB | Peak: {m:7.1f} MB | Reserved: {r:7.1f} MB"
            )

def main():
    # 1. Initialize Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(42)

    accelerator.print(f"=== Starting Distributed Training on {accelerator.num_processes} GPUs ===")
    accelerator.print(f"Distributed Backend: {accelerator.state.distributed_type}")

    # 2. Dataset & Tokenizer Setup (5-Class SST-5 Dataset)
    model_checkpoint = "bert-base-uncased"
    num_labels = 5
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    raw_datasets = load_dataset("SetFit/sst5")
    raw_datasets = raw_datasets.rename_column("label", "labels")

    train_subset = raw_datasets["train"].shuffle(seed=42)
    eval_subset = raw_datasets["test"].shuffle(seed=42)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    with accelerator.main_process_first():
        cols_to_remove = [col for col in train_subset.column_names if col != "labels"]
        tokenized_train = train_subset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
        tokenized_eval = eval_subset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)

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

    # 3. Model, Optimizer, and Schedule Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels
    )
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    # Prepare model, optimizer, and loaders across DDP ranks
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # 4. Training Schedule: 10 Epochs
    num_epochs = 10
    total_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_training_steps * 0.1), 
        num_training_steps=total_training_steps
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    accelerator.print(f"Total Epochs: {num_epochs} | Steps per Epoch: {len(train_dataloader)} | Total Steps: {total_training_steps}\n")

    # 5. Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0

        # Dedicated tqdm progress bar per epoch (visible only on GPU 0)
        train_bar = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch + 1:02d}/{num_epochs:02d} [Train]", 
            disable=not accelerator.is_local_main_process,
            leave=True
        )

        for step, batch in enumerate(train_bar):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.detach().float()
            current_lr = lr_scheduler.get_last_lr()[0]
            train_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        epoch_duration = time.time() - epoch_start_time
        samples_processed = len(train_subset)
        throughput = samples_processed / epoch_duration
        avg_loss = (running_loss / len(train_dataloader)).item()

        # Multi-GPU Evaluation with Eval tqdm progress bar
        model.eval()
        eval_bar = tqdm(
            eval_dataloader, 
            desc=f"Epoch {epoch + 1:02d}/{num_epochs:02d} [Eval] ", 
            disable=not accelerator.is_local_main_process,
            leave=False
        )

        for batch in eval_bar:
            with torch.no_grad():
                outputs = model(**batch)
            
            predictions = outputs.logits.argmax(dim=-1)
            gathered_preds, gathered_labels = accelerator.gather_for_metrics(
                (predictions, batch["labels"])
            )
            accuracy_metric.add_batch(predictions=gathered_preds, references=gathered_labels)
            f1_metric.add_batch(predictions=gathered_preds, references=gathered_labels)

        eval_acc = accuracy_metric.compute()["accuracy"]
        eval_f1 = f1_metric.compute(average="macro")["f1"]

        accelerator.wait_for_everyone()
        log_gpu_memory(accelerator, step_name=f"Epoch {epoch + 1}/{num_epochs}")

        accelerator.print(
            f"--> Epoch {epoch + 1:02d}/{num_epochs:02d} Summary: "
            f"Accuracy = {eval_acc * 100:.2f}% | "
            f"Macro F1 = {eval_f1:.4f} | "
            f"Avg Loss = {avg_loss:.4f} | "
            f"Throughput = {throughput:.2f} samples/sec | "
            f"Duration = {epoch_duration:.2f}s\n"
        )

    accelerator.print("=== Distributed Training Finished Successfully ===")
    accelerator.end_training()

if __name__ == "__main__":
    main()
