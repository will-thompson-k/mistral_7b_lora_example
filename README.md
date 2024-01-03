# mistral_7b_lora_example

A straightforward example illustrating how to fine-tune Mistral-7B with [QLoRA](https://arxiv.org/abs/2305.14314).

Inspired by [this blogpost](https://blog.neuralwork.ai/an-llm-fine-tuning-cookbook-with-mistral-7b/),
which borrowed from [this QLoRA notebook](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing).

**Work In Progress**.

Uses huggingface, trl, peft, bitsandbytes and pytorch (obviously).

Install environment using `poetry` then run `poetry run python train.py` to SFT.

**Note**: Requires (Nvidia) GPUs to run.