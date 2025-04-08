from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

def prepare_model(model_name="gpt2"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT2にはpad_tokenがないので明示的に設定
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    return model, tokenizer

def train_model(model, tokenizer):
    # 小規模データセットで検証（本番はRAG用に差し替えてOK）
    dataset = load_dataset("yelp_polarity", split="train[:1%]")

    def tokenize(example):
        tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()  # 正しい形でのラベル指定（再発防止）
        return tokens

    tokenized = dataset.map(tokenize, batched=True)

    # DataCollatorでパディング処理＆ラベル整合性を保証（これがLoRAとの相性◎）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPTはCausal LMなのでMLM=Falseが正解
    )

    training_args = TrainingArguments(
        output_dir="./lora_results",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
