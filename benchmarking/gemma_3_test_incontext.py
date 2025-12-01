import pandas as pd
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import os, torch
import editdistance

#this function assumes that the train and test set data have been created using the preprocess.py function
def get_train_path(test_set_folder, test_set_file):
    train_set_folder = test_set_folder.replace('test', 'train')
    all_files = os.listdir(train_set_folder)
    min_dist = 1000000
    target_file = ""
    for f in all_files:
        dist = editdistance.eval(f, test_set_file)
        if dist < min_dist:
            target_file = f
            min_dist = dist
    target_file_path = os.path.join(train_set_folder, target_file)
    df = pd.read_csv(target_file_path, sep='\t')
    return df

def add_examples(df_sample_dataset, user_prompt, setting, div):
    prompt = f"The following are examples of questions (with answers) about nutrition.\n\n"

    # Create few-shot prompts
    if setting == 'one-shot':
        n = 1
    elif setting == 'five-shot':
        n = 5
    if div:
        df_sample_dataset = df_sample_dataset[df_sample_dataset['instruction'].str.contains("</div>")==True]
        if 'FoodOn ontology' in user_prompt:
            df_sample_dataset = df_sample_dataset[df_sample_dataset['instruction'].str.contains("FoodOn ontology") == True]
        if 'SNOMEDCT ontology' in user_prompt:
            df_sample_dataset = df_sample_dataset[df_sample_dataset['instruction'].str.contains('SNOMEDCT ontology') == True]
        if 'Hansard taxonomy' in user_prompt:
            df_sample_dataset = df_sample_dataset[df_sample_dataset['instruction'].str.contains('Hansard taxonomy') == True]
    else:
        df_sample_dataset = df_sample_dataset[df_sample_dataset['instruction'].str.contains("</div>")==False]

    df_sample = df_sample_dataset.sample(n=n)
    df_sample["sample"] = df_sample.apply(lambda x: "Question: " + x["instruction"] + '\nAnswer: ' + x['output'], axis=1)
    for idx, row in df_sample.iterrows():
        prompt += row['sample'] + '\n\n'

    prompt += "Respond to the following question in the same manner as seen in the examples above.\n\nQuestion: " + user_prompt + '\nAnswer: '
    return prompt



if __name__ == '__main__':
    #change path to test sets if needed
    TEST_SET_PATH = 'test_sets_all_data'
    model_id = "google/gemma-3-4b-it"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    settings = ['zero-shot', 'one-shot', 'five-shot']
    folds = ['split_0', 'split_1', 'split_2', 'split_3', 'split_4']
    test_batch = 24
    for setting in settings:
        for fold in folds:
            # Loading a dataset
            print('Fold: ', fold)
            test_folder = os.path.join(TEST_SET_PATH, fold)

            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", attn_implementation="eager"
            ).eval()
            model = model.bfloat16()

            processor = AutoProcessor.from_pretrained(model_id)

            all_datasets = os.listdir(test_folder)
            for td in all_datasets:
                all_data = []
                if '_combined.tsv' not in td:
                    print('Testing on ', td)
                    df_sample_dataset = get_train_path(test_folder, td)
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
                                div = False
                                if setting in ['one-shot', 'five-shot']:
                                    user_prompt = add_examples(df_sample_dataset, user_prompt, setting, div)
                                get_answer = True
                            else:
                                div = True
                                _, question = user_prompt.split("</div>")
                                if setting in ['one-shot', 'five-shot']:
                                    user_prompt = add_examples(df_sample_dataset, prev_answer.strip() + " " + question.strip(), setting, div)

                            messages = [
                                {
                                    "role": "system",
                                    "content": [{"type": "text", "text": f"{system_prompt}".strip()}]
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": f"{user_prompt}".strip()}
                                    ]
                                }
                            ]

                            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            tokenizer_input.append(prompt)

                        inputs = processor(text=tokenizer_input, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
                        with torch.inference_mode():
                            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                        answers = processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        answers = [(" ".join((x.split('</s>')[0].split('<|end_of_text|>')[0].split('<|im_end|>')[0]).split('<|eot_id|>')[0].split())).strip() for x in answers]
                        if get_answer:
                            prev_answer = answers[0]

                        tokenizer_input = [" ".join(x.strip().split()) for x in tokenizer_input]

                        original_prompts = [x.strip() for x in original_prompts]
                        batched_examples = zip(original_prompts, tokenizer_input, answers, true_outputs)
                        all_data.extend(batched_examples)

                    suffix = '_gemma-3-4b-it/'

                    df = pd.DataFrame(all_data, columns=['Original prompt', 'True prompt', 'Answer', 'True'])
                    if not os.path.exists("results/" + setting + suffix):
                        os.makedirs("results/" + setting + suffix)
                    results_output = "results/" + setting + suffix + td.split('.')[0] + '_' + fold + ".tsv"
                    df.to_csv(results_output, encoding='utf8', sep='\t', index=False)
            del model
            del processor



