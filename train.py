""" This is a simple (commented) example of how to fine-tune (train) Mistral-7b."""

from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import torch

# download dataset (from blog)
DATASET = load_dataset("neuralwork/fashion-style-instruct")
# model_id to fine-tune
MODEL_ID = "mistralai/Mistral-7B-v0.1"
# output of SFT, QLoRA layers
OUTPUT_DIR = "model_output"


# borrowing the neuralwork blog's SFT dataset on fashion.
def format_instruction(sample):
    return f"""You are a personal stylist recommending fashion \
    advice and clothing combinations. Use the self body and \
    style description below, combined with the event described in the \
    context to generate 5 self-contained and complete outfit combinations.\
        ### Input:
        {sample["input"]}

        ### Context:
        {sample["context"]}

        ### Response:
        {sample["completion"]}
    """


def generate_qlora_model_config():
    """
    QLoRA model configuration.

    options are:
        (1) QLoRA Int8
        (2) QLoRA FP4
        (3) QLoRA NF4 + DQ

    We chose (3) since QLoRA paper states:

    Overall, NF4 with double quantization (DQ) matches BFloat16
    performance, while FP4 is consistently one percentage point behind both.
    """
    return BitsAndBytesConfig(
        # flag is used to enable 4-bit quantization by replacing the Linear
        # layers with FP4/NF4 layers from bitsandbytes
        load_in_4bit=True,
        # This flag is used for nested quantization where the quantization
        # constants from the first quantization are quantized again
        bnb_4bit_use_double_quant=True,
        # fp4 or nf4 for 4 bit quantization
        bnb_4bit_quant_type="nf4",
        # speed up by cutting from 32->16 at load time
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


bnb_config = generate_qlora_model_config()

# load tokenizer, pad samples with end of sentence token
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# true for llama-2 and (believe) it's true for Mistral
tokenizer.pad_token = tokenizer.eos_token

# load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, use_cache=False, device_map="auto"
)
# sanity check this is 1 per
# https://github.com/huggingface/transformers/issues/25137
model.config.pretraining_tp = 1

# (Q)LoRA training config
peft_config = LoraConfig(
    r=32,  # As bigger the R bigger the parameters to train.
    # a scaling factor that adjusts the magnitude of the weight matrix.
    # It seems that as higher more weight have the new training.
    lora_alpha=64,
    # If targeting all linear layers, else if just attention [q_proj, k_proj]
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",  # this specifies if the bias parameter should be trained.
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


# originally from:
#  https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print_trainable_parameters(model)

# taken directly from blog post, have not rigorously evaluated
model_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=1,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    warmup_steps=2,
)

# Supervised Fine-Tuning Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=DATASET["train"],
    peft_config=peft_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=model_args,
)

# train
trainer.train()

# save model to output_dir in TrainingArguments
trainer.save_model()
