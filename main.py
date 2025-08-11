import os
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator

config = AutoConfig.for_model(
    "codegen",
    vocab_size=50400,
    n_positions=2048,
    n_ctx=2048,
    n_embd=2048,
    n_layer=20,
    n_head=16,
    rotary_dim=64,
    n_inner=None, 
    activation_function="gelu_new",
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    use_cache=True,
    bos_token_id=50256,
    eos_token_id=50256,
    tie_word_embeddings=False,
)

model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")  

permissive_licenses = [
    "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "cc0-1.0",
    "unlicense", "isc", "artistic-2.0", "epl-2.0", "mpl-2.0"
]

def load_and_filter_dataset(languages=None):
    if languages is None:
        languages = ["c", "c++", "c-sharp", "dart", "go", "java", "javascript", "kotlin", "lua", "php", "python", "ruby", "rust", "scala", "shell", "sql", "swift", "typescript", "vue"]
    datasets_list = []
    for lang in languages:
        ds = load_dataset("bigcode/the-stack-dedup", split="train", data_dir=lang, streaming=True)
        # Filter for permissive licenses
        filtered_ds = ds.filter(lambda example: example["license"] in permissive_licenses if "license" in example else False)
        datasets_list.append(filtered_ds)
    full_ds = concatenate_datasets(datasets_list)
    return full_ds

dataset = load_and_filter_dataset()  

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["content"]], truncation=True, max_length=1024)

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset",
)

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // 1024) * 1024  
    result = {
        k: [t[i : i + 1024] for i in range(0, total_length, 1024)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    desc="Grouping texts",
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./codegen2-1b-scratch",
    overwrite_output_dir=True,
    num_train_epochs=1,  
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=8,  
    learning_rate=5e-5,
    weight_decay=0.01,
    max_steps=-1,  
    save_steps=10000,
    logging_steps=100,
    fp16=True,  
    report_to="tensorboard",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./codegen2-1b-scratch/final")
tokenizer.save_pretrained("./codegen2-1b-scratch/final")
