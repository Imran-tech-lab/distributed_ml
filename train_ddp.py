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

def print_table_header(accelerator):
    separator = "+-------+------------+-----------+----------+----------+----------+-------------+-------------+----------+"
    header    = "| Epoch | Train Loss | Train Acc | Val Loss | Val Acc  | Macro F1 | GPU 0 Peak  | GPU 1 Peak  | Duration |"
    accelerator.print(separator)
    accelerator.print(header)
    accelerator.print(separator)

def print_table_row(accelerator, epoch, tr_loss, tr_acc, val_loss, val_acc, f1, g0_mem, g1_mem, duration):
    row = (
        f"| {epoch:^5} | {tr_loss:^10.4f} | {tr_acc * 100:^8.2f}% | {val_loss:^8.4f} | "
        f"{val_acc * 100:^7.2f}% | {f1:^8.4f} | {g0_mem:^11} | {g1_mem:^11} | {duration:^6.1f}s |"
    )
    accelerator.print(row)

def print_table_footer(accelerator):
    separator = "+-------+------------+-----------+----------+----------+----------+-------------+-------------+----------+"
    accelerator.print(separator)

def get_peak_memory_per_gpu(accelerator):
    """Collects peak VRAM across all DDP ranks."""
    device = accelerator.device
    torch.cuda.synchronize(device)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    gathered_peaks = accelerator.gather(torch.tensor([peak_mb], device=device, dtype=torch.float32))
    peaks = gathered_peaks.cpu().numpy()
    
    gpu0_str = f"{peaks[0]:.0f} MB"
    gpu1_str = f"{peaks[1]:.0f} MB" if len(peaks) > 1 else "N/A"
    return gpu0_str, gpu1_str

def main():
    # 1. Initialize Accelerator (FP16 enabled for dual T4 Tensor Cores)
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(42)

    accelerator.print(f"\n[DDP] Initialized on {accelerator.num_processes}x GPUs ({accelerator.state.distributed_type})")

    # 2. Upgrade to RoBERTa and extend sequence length to 256
    model_checkpoint = "roberta-base"
    num_labels = 5
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    raw_datasets = load_dataset("Yelp/yelp_review_full")
    raw_datasets = raw_datasets.rename_column("label", "labels")

    # 10k train, 2.5k evaluation samples
    train_subset = raw_datasets["train"].shuffle(seed=42).select(range(10000))
    eval_subset = raw_datasets["test"].shuffle(seed=42).select(range(2500))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    with accelerator.main_process_first():
        cols_to_remove = [col for col in train_subset.column_names if col != "labels"]
        tokenized_train = train_subset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
        tokenized_eval = eval_subset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Batch size 32 per GPU -> Global batch size = 64
    batch_size = 32
    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_eval, batch_size=batch_size, collate_fn=data_collator)

    # 3. Model & Optimizer Setup
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    
    # 1.5e-5 lr with higher weight decay (0.02) to mitigate overfitting
    optimizer = AdamW(model.parameters(), lr=1.5e-5, weight_decay=0.02)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_epochs = 6  # 5-6 epochs is standard to reach peak performance without severe overfit
    total_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_training_steps * 0.1), 
        num_training_steps=total_training_steps
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    accelerator.print(f"[Config] Model: {model_checkpoint} | Max Length: 256 | Epochs: {num_epochs} | Global Batch Size: {batch_size * accelerator.num_processes}\n")
    print_table_header(accelerator)

    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        running_train_loss = 0.0
        train_correct_tensor = torch.tensor(0, device=accelerator.device, dtype=torch.long)
        train_total_tensor = torch.tensor(0, device=accelerator.device, dtype=torch.long)

        train_bar = tqdm(
            train_dataloader, 
            desc=f"Training Epoch {epoch + 1:02d}", 
            disable=not accelerator.is_local_main_process,
            leave=False
        )

        for step, batch in enumerate(train_bar):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            running_train_loss += loss.detach().float()
            
            # Local batch predictions
            preds = outputs.logits.argmax(dim=-1)
            train_correct_tensor += (preds == batch["labels"]).sum()
            train_total_tensor += batch["labels"].size(0)

            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = (running_train_loss / len(train_dataloader)).item()

        # Gather training accuracy across all GPUs
        global_train_correct = accelerator.gather(train_correct_tensor).sum().item()
        global_train_total = accelerator.gather(train_total_tensor).sum().item()
        train_acc = global_train_correct / global_train_total

        # 5. Distributed Validation Pass
        model.eval()
        total_eval_loss = 0.0
        total_eval_samples = 0

        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            
            # Loss synchronization
            batch_sz = batch["labels"].size(0)
            gathered_loss = accelerator.gather(outputs.loss * batch_sz).sum()
            gathered_count = accelerator.gather(torch.tensor(batch_sz, device=accelerator.device)).sum()

            total_eval_loss += gathered_loss.item()
            total_eval_samples += gathered_count.item()

            # Prediction gathering for global metrics
            predictions = outputs.logits.argmax(dim=-1)
            gathered_preds, gathered_labels = accelerator.gather_for_metrics(
                (predictions, batch["labels"])
            )
            accuracy_metric.add_batch(predictions=gathered_preds, references=gathered_labels)
            f1_metric.add_batch(predictions=gathered_preds, references=gathered_labels)

        avg_val_loss = total_eval_loss / total_eval_samples
        val_acc = accuracy_metric.compute()["accuracy"]
        val_f1 = f1_metric.compute(average="macro")["f1"]

        gpu0_peak, gpu1_peak = get_peak_memory_per_gpu(accelerator)

        # Print consolidated table row
        print_table_row(
            accelerator,
            epoch=epoch + 1,
            tr_loss=avg_train_loss,
            tr_acc=train_acc,
            val_loss=avg_val_loss,
            val_acc=val_acc,
            f1=val_f1,
            g0_mem=gpu0_peak,
            g1_mem=gpu1_peak,
            duration=epoch_duration
        )

    print_table_footer(accelerator)
    accelerator.print("\n=== Distributed Training Complete ===")
    accelerator.end_training()

if __name__ == "__main__":
    main()
