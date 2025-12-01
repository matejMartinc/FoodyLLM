import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import os, torch
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

#This function concatenates all the train datasets for a specific fold and adds the dolly_hhrlhf data to the train set
#to prevent model overfitting
def concat_all(folder_path, adapter_model, shuffle=False, add_data=True):
    df_all = pd.DataFrame([], columns=['instruction', 'output'])
    files = os.listdir(folder_path)
    for f in files:
        if '_combined.tsv' not in f:
            path = os.path.join(folder_path, f)
            df = pd.read_csv(path, encoding='utf8', sep='\t')
            df_all = pd.concat([df_all, df])
    print('Data combined')
    if add_data:
        dataset = load_dataset("mosaicml/instruct-v3")
        dataset = dataset.filter(lambda x: x["source"] == "dolly_hhrlhf")
        train_mosaic = dataset["train"]#.select(range(len(df_all)))
        train = []
        for example in train_mosaic:
            original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            user_prompt = example['prompt'].replace(original_system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
            true_output = example['response']
            user_prompt = user_prompt.replace('�', ' ').replace('음', ' ')
            true_output = true_output.replace('�', ' ').replace('음', ' ')
            if len(user_prompt.split()) + len(true_output.split()) < 1024:
                train.append((user_prompt.strip(), true_output.strip()))
        df_mosaic = pd.DataFrame(train, columns=['instruction', 'output'])
        df_mosaic.to_csv('data/mosaic_whole_long_removed.tsv', encoding='utf8', sep='\t', index=False)
        df_all = pd.concat([df_all, df_mosaic])
        print('Added mosaic data')
    if shuffle:
        df_all = df_all.sample(frac=1, random_state=123).reset_index(drop=True)
    df_all.to_csv(os.path.join(folder_path, adapter_model) + '_combined.tsv', encoding='utf8', index=False, sep='\t')


if __name__ == '__main__':
    #change the paths to data if needed
    TRAIN_SET_PATH = 'train_sets_all_data'
    TEST_SET_PATH = 'test_sets_all_data'
    base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setting this to True will only test the already trained models and the script will not train new models
    load_adapter = False
    print('Load adapter', load_adapter)
    adapter_model = "foodyLLM-Meta-LLama-3-8B-Instruct"
    folds = ['split_0', 'split_1', 'split_2', 'split_3', 'split_4']
    test_batch = 16
    for fold in folds:
        # Loading a dataset
        print('Fold: ', fold)
        train_folder = os.path.join(TRAIN_SET_PATH, fold)
        test_folder = os.path.join(TEST_SET_PATH, fold)
        concat_all(train_folder, adapter_model, shuffle=True, add_data=True)
        train_path = os.path.join(train_folder, adapter_model) + '_combined.tsv'

        # Load base model(Mistral 7B)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map={"": 0},
            attn_implementation="eager"
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        #set pad token which does by default not exist in the LLama 3 model
        tokenizer.pad_token = '<|pad|>'
        tokenizer.pad_token_id = 128255

        # Importing the dataset
        dataset = load_dataset("csv", sep='\t', data_files={"train": train_path})

        def format_chat_template(row):
            row_json = [{"role": "user", "content": row["instruction"]},
                        {"role": "assistant", "content": row["output"]}]
            row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
            row["text"] = row['text'].replace('</div>', '').strip()
            return row

        dataset = dataset.map(
            format_chat_template,
            num_proc=4,
        )

        if load_adapter:
            model.load_adapter("models/" + adapter_model + '_' + fold)
        else:
            model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
            model.config.pretraining_tp = 1
            model = prepare_model_for_kbit_training(model)
            # LoRA config
            peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
            )
            model = get_peft_model(model, peft_config)

            # Training Arguments
            # Hyperparameters should be adjusted based on the hardware you using
            training_arguments = SFTConfig(
                output_dir="./results/" + adapter_model,
                num_train_epochs=1,
                per_device_train_batch_size=10,
                gradient_accumulation_steps=2,
                max_seq_length=1024,
                optim="paged_adamw_8bit",
                save_steps=5000,
                logging_steps=30,
                learning_rate=2e-4,
                fp16=False,
                bf16=False,
                warmup_steps=10,
                group_by_length=True,
                dataset_text_field="text",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )


            trainer = SFTTrainer(
                model,
                train_dataset=dataset["train"],
                peft_config=peft_config,
                tokenizer=tokenizer,
                args=training_arguments,
                packing=False,
            )

            trainer.train()
            # Save the fine-tuned model
            trainer.model.save_pretrained("models/" + adapter_model + '_' + fold)

            del trainer

        model.config.use_cache = True
        model.eval()
        all_datasets = os.listdir(test_folder)

        #Test the model on each dataset in the test fold
        for td in all_datasets:
            all_data = []
            if '_combined.tsv' not in td:
                print('Testing on ', td)
                if 'ner' not in td:
                    current_batch = test_batch
                else:
                    current_batch = 1
                df = pd.read_csv(os.path.join(test_folder, td), encoding='utf8', sep='\t')
                prev_answer = ""
                for i in range(0, len(df), current_batch):
                    if i % (current_batch * 10) == 0:
                        print('Generating example', i)
                    examples = df[i:i + current_batch]
                    tokenizer_input = []
                    true_outputs = []
                    for true_output in examples['output']:
                        true_outputs.append(true_output)
                    original_prompts = []
                    true_prompts = []
                    for user_prompt in examples['instruction']:
                        original_prompts.append(user_prompt)
                        system_prompt = ''
                        get_answer = False
                        if not '</div>' in user_prompt:
                            messages = [
                                {
                                    "role": "user",
                                    "content": f"{system_prompt} {user_prompt}".strip()
                                }
                            ]
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            get_answer = True
                        else:
                            _, question = user_prompt.split("</div>")
                            messages = [
                                {
                                    "role": "user",
                                    "content": f"{prev_answer.strip()} {question.strip()}".strip()
                                }
                            ]
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        tokenizer_input.append(prompt)

                    inputs = tokenizer(tokenizer_input, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
                    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True)
                    answers = tokenizer.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:])
                    answers = [(" ".join((x.split('<|end_of_text|>')[0].split('<|im_end|>')[0]).split('<|eot_id|>')[0]
                                .replace("<|start_header_id|>assistant", '').replace("<|end_header_id|>", '')
                                         .replace("<|start_header_id|>", '').split())).strip() for x in answers]
                    if get_answer:
                        prev_answer = answers[0]
                    tokenizer_input = [" ".join(x.strip().split()) for x in tokenizer_input]
                    original_prompts = [x.strip() for x in original_prompts]
                    batched_examples = zip(original_prompts, tokenizer_input, answers, true_outputs)
                    all_data.extend(batched_examples)

                df = pd.DataFrame(all_data, columns=['Original prompt', 'True prompt', 'Answer', 'True'])
                results_output = "results/" + adapter_model + '/' + td.split('.')[0] + '_' + fold + ".tsv"
                df.to_csv(results_output, encoding='utf8', sep='\t', index=False)
        del model
        del tokenizer



