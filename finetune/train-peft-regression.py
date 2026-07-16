import os
import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
from datetime import datetime

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# -----------------------------
# Arguments
# -----------------------------
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")

    # Unified switch
    peft_method: str = field(
        default="none",
        metadata={"help": "PEFT method: one of {none, lora, qlora, adalora}"}
    )

    # LoRA / QLoRA args
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="Wqkv", metadata={"help": "where to perform LoRA"})

    # AdaLoRA args
    adalora_init_r: int = field(default=12, metadata={"help": "initial hidden dimension for AdaLoRA"})
    adalora_target_r: int = field(default=8, metadata={"help": "target hidden dimension for AdaLoRA"})
    adalora_alpha: int = field(default=32, metadata={"help": "alpha for AdaLoRA"})
    adalora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for AdaLoRA"})
    adalora_target_modules: str = field(default="Wqkv", metadata={"help": "modules to apply AdaLoRA"})

    total_step: int = field(default=160000, metadata={"help": "total training steps"})
    tinit: int = field(default=16000, metadata={"help": "warmup steps before rank adaptation starts"})
    tfinal: int = field(default=140000, metadata={"help": "end step of adaptation"})
    deltaT: int = field(default=1000, metadata={"help": "frequency of rank updates"})
    beta1: float = field(default=0.85, metadata={"help": "beta1 for adaptive rank"})
    beta2: float = field(default=0.85, metadata={"help": "beta2 for adaptive rank"})
    orth_reg_weight: float = field(default=0.1, metadata={"help": "weight for orthogonal regularization"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to data directory with train/dev/test CSV"})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 = disabled"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)


# -----------------------------
# Utils
# -----------------------------
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def get_alter_of_dna_sequence(sequence: str):
    """Get complement of DNA sequence"""
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([MAP[c] for c in sequence])


def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])


def _dist_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """
    Load or generate k-mer string for each DNA sequence. The generated 
    k-mer string will be saved to the same directory as the original 
    data with the same name but with a suffix of "_{k}mer".
    """
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning("Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
    return kmer


# -----------------------------
# Dataset
# -----------------------------
class SupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    """
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]  # skip header

        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [float(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [float(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            # only write file on the first process
            if _dist_ready():
                rank = torch.distributed.get_rank()
                if rank not in [0, -1]:
                    torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if _dist_ready() and torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.tensor(labels).float()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

# -----------------------------
# Metrics
# -----------------------------
from scipy.stats import pearsonr, spearmanr

def calculate_regression_metrics(predictions: np.ndarray, labels: np.ndarray):
    # Filter out ignored labels if any
    valid_mask = labels != -100
    preds = predictions[valid_mask]
    targets = labels[valid_mask]

    mse = sklearn.metrics.mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae = sklearn.metrics.mean_absolute_error(targets, preds)
    mape = sklearn.metrics.mean_absolute_percentage_error(targets, preds)
    r2 = sklearn.metrics.r2_score(targets, preds)

    # Correlation metrics
    try:
        pearson = pearsonr(targets, preds)[0]
    except Exception:
        pearson = 0.0

    try:
        spearman = spearmanr(targets, preds)[0]
    except Exception:
        spearman = 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }


def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    return logits


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.squeeze(logits)
    labels = np.squeeze(labels)
    
    if np.random.rand() < 0.1:
        plot_predictions_vs_truth(predictions, labels)
        plot_label_distribution(labels)
        plot_residuals(predictions, labels)
        print("Plotting predictions vs labels...")

    return calculate_regression_metrics(predictions, labels)

def plot_predictions_vs_truth(preds, labels):
    plt.figure(figsize=(6, 6))
    plt.scatter(labels, preds, alpha=0.3)

    min_val = min(labels.min(), preds.min())
    max_val = max(labels.max(), preds.max())

    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel("True values")
    plt.ylabel("Predictions")
    plt.title("Predictions vs Ground Truth")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"pred-vs-label_{timestamp}.png")
    plt.close()


def plot_label_distribution(labels):
    plt.figure()
    plt.hist(labels, bins=100)
    plt.title("Label distribution")
    plt.savefig(f"label-distribution_{np.random.randint(10000)}.png")
    plt.close()


def plot_residuals(preds, labels):
    residuals = preds - labels

    plt.figure()
    plt.scatter(preds, residuals, alpha=0.3)
    plt.axhline(0, linestyle='--')

    plt.xlabel("True values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Prediction")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"residuals_{timestamp}.png")
    plt.close()


# -----------------------------
# Train
# -----------------------------
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    peft_method = model_args.peft_method.lower().strip()
    if peft_method not in {"none", "lora", "qlora", "adalora"}:
        raise ValueError(f"Invalid peft_method={model_args.peft_method}. Use one of: none, lora, qlora, adalora")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "train.csv"),
        kmer=data_args.kmer
    )
    val_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "dev.csv"),
        kmer=data_args.kmer
    )
    test_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "test.csv"),
        kmer=data_args.kmer
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.pad_token_id = tokenizer.pad_token_id
    config.num_labels = 1
    config.problem_type = "regression"

    # Load model differently for qlora
    if peft_method == "qlora":
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            config=config,
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        model.config.use_cache = False
        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False
        )
        
        if hasattr(model, "bert") and hasattr(model.bert, "pooler"):
            old_dense = model.bert.pooler.dense

            new_dense = torch.nn.Linear(
                old_dense.in_features,
                old_dense.out_features,
                bias=True
            )

            new_dense.weight.data = old_dense.weight.data.float()
            new_dense.bias.data = old_dense.bias.data.float()

            model.bert.pooler.dense = new_dense.to("cuda")
            
        if hasattr(model, "classifier"):
            old = model.classifier
            new = torch.nn.Linear(old.in_features, old.out_features, bias=True)
            new.weight.data = old.weight.data.float()
            new.bias.data = old.bias.data.float()
            model.classifier = new.to("cuda")
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!11
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            config=config,
            trust_remote_code=True,
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply PEFT
    if peft_method in {"lora", "qlora"}:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elif peft_method == "adalora":
        adalora_config = AdaLoraConfig(
            init_r=model_args.adalora_init_r,
            target_r=model_args.adalora_target_r,
            lora_alpha=model_args.adalora_alpha,
            target_modules=list(model_args.adalora_target_modules.split(",")),
            lora_dropout=model_args.adalora_dropout,
            total_step=model_args.total_step,
            tinit=model_args.tinit,
            tfinal=model_args.tfinal,
            deltaT=model_args.deltaT,
            beta1=model_args.beta1,
            beta2=model_args.beta2,
            orth_reg_weight=model_args.orth_reg_weight,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, adalora_config)
        model.print_trainable_parameters()

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    train()