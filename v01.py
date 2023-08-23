!pip install bitsandbytes
!pip install torch
!pip install transformers
!pip install accelerate
!pip install scipy
!pip install peft
!pip install datasets
!pip install wandb




import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


####  https://github.com/huggingface/peft

model_name = "EleutherAI/gpt-neox-20b"

#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map={"":0})


model.gradient_checkpointing_enable()


from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)



from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

#data = load_dataset("Anthropic/hh-rlhf")
#data = data.map(lambda samples: tokenizer(samples["chosen"]), batched=True)

import transformers

tokenizer.pad_token = tokenizer.eos_token
#tokenizer.truncation = True

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=16,
        max_steps=40,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()





while True:
    text = input("Enter prompt here:")
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
