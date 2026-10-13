# Distributed Emotion Classification (PyTorch DDP)

Fine-tuning `roberta-base` on the 6-class `dair-ai/emotion` dataset using **Distributed Data Parallelism (DDP)** across **2x NVIDIA Tesla T4 GPUs** on Kaggle with Hugging Face `accelerate`.

---

## Quickstart

```bash
git clone https://github.com/Imran-tech-lab/distributed_ml.git
cd distributed_ml
pip install -q -r requirements.txt

```

### Run Distributed Training (2x T4 GPUs)

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=no \
    train_ddp.py

```

---

## Key Features

* **Multi-GPU DDP:** Gradient synchronization via NCCL and FP16 mixed precision across 2 GPUs.
* **Class Balancing:** Inverse class weighting injected into Cross-Entropy Loss to resolve minority class collapse (`surprise`, `love`).
* **Telemetry Gathering:** Synchronized, cross-process GPU memory tracking (`torch.distributed.all_gather_object`).
* **Early Stopping:** Automatically saves and restores the best checkpoint based on validation Macro F1 score.

---

## Results

### Training Metrics (Dual T4)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Macro F1 | GPU 0 Peak | GPU 1 Peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | 1.3198 | 47.27% | 0.4623 | 78.55% | 0.7727 | 3018 MB | 3228 MB |
| **2** | 0.3812 | 86.46% | 0.2418 | 90.90% | 0.8885 | 3018 MB | 3228 MB |
| **3** | 0.2499 | 90.83% | 0.2235 | 91.15% | 0.8904 | 3230 MB | 3228 MB |

* **Test Accuracy:** **90.65%**
* **Test Macro F1:** **0.8701**

---

## Sample Inference

```python
=================================================================
DEMO INFERENCE ON NEW UNSEEN TEXT
=================================================================
Input : "I am so proud of my sister for graduating with honors today!"
Output: Emotion = JOY        | Confidence: 98.91%

Input : "I feel completely crushed, empty, and hopeless after that conversation."
Output: Emotion = SADNESS    | Confidence: 98.85%

Input : "I can't believe they broke their word again, it makes my blood boil!"
Output: Emotion = SADNESS    | Confidence: 89.52%

Input : "Hearing footsteps outside my door in the dark made my heart race with panic."
Output: Emotion = FEAR       | Confidence: 98.96%

Input : "Wait, you actually won first place? I never saw that coming at all!"
Output: Emotion = JOY        | Confidence: 77.00%

Input : "Holding your hand makes me realize how deeply I cherish you in my life."
Output: Emotion = LOVE       | Confidence: 97.16%

```
