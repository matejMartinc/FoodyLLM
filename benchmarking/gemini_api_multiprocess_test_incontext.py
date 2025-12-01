import nest_asyncio

nest_asyncio.apply()

from google import genai

GOOGLE_API_KEY = ""

import asyncio
from tqdm import tqdm
from aiolimiter import AsyncLimiter
import os
import editdistance
import pandas as pd


api_rate_limiter = AsyncLimiter(max_rate=10, time_period=1)

MAX_RETRIES = 3

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


def add_examples(df_sample_dataset, user_prompt, setting):
    prompt = f"The following are examples of questions (with answers) about nutrition.\n\n"

    # Create few-shot prompts
    if setting == 'one-shot':
        n = 1
    elif setting == 'five-shot':
        n = 5

    df_sample = df_sample_dataset.sample(n=n)
    df_sample["sample"] = df_sample.apply(lambda x: "Question: " + x["instruction"] + '\nAnswer: ' + x['output'], axis=1)
    for idx, row in df_sample.iterrows():
        prompt += row['sample'] + '\n\n'

    prompt += "Respond to the following question in the same manner as seen in the examples above.\n\nQuestion: " + user_prompt + '\nAnswer: '
    return prompt




async def process_document(i, document, setting, df_sample_dataset, delay=1):
    """
    Process a single document: wait for a given delay, then send the prompt to the Gemini model,
    enforcing the API rate limit with a limiter.

    After receiving the response, the function attempts to clean and evaluate the text.
    If that fails, it retries the LLM call up to MAX_RETRIES times.
    """

    client = genai.Client(api_key=GOOGLE_API_KEY)
    original_prompt = document['instruction']
    true_output = document['output']

    if setting in ['one-shot', 'five-shot']:
        prompt = add_examples(df_sample_dataset, original_prompt, setting)
    else:
        prompt = original_prompt

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            await asyncio.sleep(delay)
            async with api_rate_limiter:
                result = await client.aio.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=[
                        prompt,
                    ]
                )
            answer = result.text

            return i, answer, original_prompt, prompt, true_output

        except Exception as e:
            print(e)
            attempt += 1
            if attempt >= MAX_RETRIES:
                return i, "Error in response", original_prompt, prompt, true_output


async def process_grouped_documents(documents_grouped, output_path, setting, df_sample_dataset, current_chunk):
    """
    Processes all documents in parallel by creating tasks for each document.
    documents_grouped is a list of tuples (original_index, document).
    Uses asyncio.as_completed to yield tasks as they finish, updating a progress bar.
    Returns a dictionary mapping the original document indices to their generated text.
    """

    tasks = [
        asyncio.create_task(process_document(i, row, setting, df_sample_dataset))
        for i, (_, row) in enumerate(documents_grouped.iterrows())
    ]
    output_file = open(output_path, 'a')

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Documents"):
        all_output = await future
        i, answer, original_prompt, prompt, true_output = all_output
        try:
            answer = " ".join(answer.split()).strip()
        except:
            answer = "Error in response"
        prompt = " ".join(prompt.split()).strip()
        original_prompt = " ".join(original_prompt.split()).strip()
        true_output = " ".join(true_output.split()).strip()
        output_file.write(str(current_chunk + i) + '\t' + str(original_prompt) + '\t' + str(prompt) + '\t' + str(answer) + '\t' + str(true_output) + '\n')
    output_file.close()



if __name__ == '__main__':
    #change path to test sets if needed
    TEST_SET_PATH = 'test_sets_all_data'
    settings = ['zero-shot', 'one-shot', 'five-shot']
    folds = ['split_0', 'split_1', 'split_2', 'split_3', 'split_4']
    for setting in settings:
        for fold in folds:
            # Loading a dataset
            print('Fold: ', fold)
            test_folder = os.path.join(TEST_SET_PATH, fold)

            all_datasets = os.listdir(test_folder)
            for td in all_datasets:
                all_data = []
                if '_combined.tsv' not in td:
                    print('Testing on ', td)
                    df_sample_dataset = get_train_path(test_folder, td)
                    #if td == 'nel_bootstrap_samples_fcd_mistral_hansard.tsv':
                    if 'ner' not in td:
                        df = pd.read_csv(os.path.join(test_folder, td), encoding='utf8', sep='\t')
                        if not os.path.exists("results/" + setting + '_gemini_2.0_flash/'):
                            os.makedirs("results/" + setting + '_gemini_2.0_flash/')
                        output_path = "results/" + setting + '_gemini_2.0_flash/' + td.split('.')[0] + '_' + fold + ".tsv"
                        output_file = open(output_path, 'a')
                        output_file.write('Id\tOriginal prompt\tTrue prompt\tAnswer\tTrue\n')
                        output_file.close()
                        chunks = 1000
                        for i in range(0, len(df), chunks):
                            print("Chunk:", i)
                            asyncio.run(
                                process_grouped_documents(df[i:i + chunks], output_path, setting, df_sample_dataset, i))

    #do some additional cleaning
    folders = ['results/one-shot_gemini_2.5_flash', 'results/zero-shot_gemini_2.5_flash',
               'results/five-shot_gemini_2.5_flash']
    for folder in folders:
        files = os.listdir(folder)
        output_folder = folder + '_parsed'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for f in files:
            df = pd.read_csv(os.path.join(folder, f), sep='\t', encoding='utf8')
            df = df.sort_values(by='Id')
            df = df[['Original prompt', 'True prompt', 'Answer', 'True']]
            df['Answer'] = df['Answer'].apply(lambda x: x.replace('Answer:', '').strip())
            output_f = os.path.join(output_folder, f)
            df.to_csv(output_f, sep='\t', encoding='utf8', index=False)






