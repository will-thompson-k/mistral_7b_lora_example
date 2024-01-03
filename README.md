# mistral_7b_lora_example

A straightforward example illustrating how to fine-tune Mistal 7b with [QLoRA](https://arxiv.org/abs/2305.14314).

Derived from [this blogpost](https://blog.neuralwork.ai/an-llm-fine-tuning-cookbook-with-mistral-7b/).

<ins>*Work In Progress*</ins>.

Uses huggingface, trl, peft (via huggingface), bitsandbytes and pytorch (obviously).

Install environment using `poetry` then run `poetry run python train.py` to SFT.

**Note**: Requires (Nvidia) GPUs to run.