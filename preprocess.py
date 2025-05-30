import pandas as pd
import os
from sklearn.model_selection import KFold


TRAIN_SET_PATH = 'train_sets_all_data'
TEST_SET_PATH = 'test_sets_all_data'



def cv_split(df, output_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    for idx, split in  enumerate (kf.split(df)):
        train_df = df.iloc[split[0]]
        test_df = df.iloc[split[1]]
        if not os.path.exists(TRAIN_SET_PATH + '/' + f'split_{idx}/'):
            os.makedirs(TRAIN_SET_PATH + '/' + f'split_{idx}/')
        if not os.path.exists(TEST_SET_PATH + '/' + f'split_{idx}/'):
            os.makedirs(TEST_SET_PATH + '/' + f'split_{idx}/')
        train_df.to_csv(TRAIN_SET_PATH + '/' + f'split_{idx}/' + output_name + '.tsv', encoding='utf8', index=False, sep='\t')
        test_df.to_csv(TEST_SET_PATH + '/' + f'split_{idx}/' + output_name + '.tsv', encoding='utf8', index=False, sep='\t')


def preprocess_ner():
    datasets = ['datasets/CafeteriaFCD_instruction_response.txt',
                'datasets/CafeteriaSA_instruction_response.txt']
    for dt_path in datasets:
        output_name = dt_path.split('/')[1].replace(' ', '_').split('.')[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=123)
        with open(dt_path, 'r', encoding='utf8') as f:
            df = pd.DataFrame(f.readlines(), columns=['text'])
            for cv_idx, split in enumerate(kf.split(df)):
                train_df = df.iloc[split[0]]
                test_df = df.iloc[split[1]]
                if not os.path.exists(TRAIN_SET_PATH + '/' + f'split_{cv_idx}/'):
                    os.makedirs(TRAIN_SET_PATH + '/' + f'split_{cv_idx}/')
                if not os.path.exists(TEST_SET_PATH + '/' + f'split_{cv_idx}/'):
                    os.makedirs(TEST_SET_PATH + '/' + f'split_{cv_idx}/')

                counter = 0
                train_dataset = []
                for line in train_df['text'].tolist():
                    if len(line.strip()) > 0:
                        chunks = line.split('[INST]')
                        counter += 1
                        prev_output = ''
                        for idx, chunk in enumerate(chunks):
                            if len(chunk.strip()) > 0:
                                instruction, output = chunk.strip().split('[/INST]')
                                if idx > 1:
                                    instruction = prev_output + ' </div> ' + instruction
                                train_dataset.append((instruction.strip(), output.strip()))
                                if idx == 1:
                                    prev_output = output
                test_dataset = []
                for line in test_df['text'].tolist():
                    if len(line.strip()) > 0:
                        chunks = line.split('[INST]')
                        counter += 1
                        prev_output = ''
                        for idx, chunk in enumerate(chunks):
                            if len(chunk.strip()) > 0:
                                instruction, output = chunk.strip().split('[/INST]')
                                if idx > 1:
                                    instruction = prev_output + ' </div> ' + instruction
                                test_dataset.append((instruction.strip(), output.strip()))
                                if idx == 1:
                                    prev_output = output
                train_df = pd.DataFrame(train_dataset, columns=['instruction', 'output'])
                train_df = train_df.sample(frac=1, random_state=123).reset_index(drop=True)
                train_df.to_csv(TRAIN_SET_PATH + '/' + f'split_{cv_idx}/' + output_name + '.tsv', encoding='utf8', index=False, sep='\t')
                test_df = pd.DataFrame(test_dataset, columns=['instruction', 'output'])
                test_df.to_csv(TEST_SET_PATH + '/' + f'split_{cv_idx}/' + output_name + '.tsv', encoding='utf8', index=False, sep='\t')


def preprocess_nel_bootstrap():
    datasets = ['datasets/FCD_foodon_instruction_response.txt',
                'datasets/FCD_hansard_instruction_response.txt',
                'datasets/FCD_snomed_ct_instruction_response.txt']

    for ds in datasets:
        dataset = []
        output_name = ds.split('/')[1].replace(' ', '_').split('.')[0]
        with open(ds, 'r', encoding='utf8') as f:
            docs = f.readlines()
            for line in docs:
                if len(line.strip()) > 0:
                    line = line.replace('[INST]', '')
                    instruction, output = line.strip().split('[/INST]')
                    dataset.append((instruction.replace(' ?', '?').strip(), output.strip()))
        df = pd.DataFrame(dataset, columns=['instruction', 'output'])
        cv_split(df, output_name)



def preprocess_USDA_FCD_and_conversion():
    datasets = [['datasets/USDA FCD mapping (synonims)/USDA_mapping.txt', 'usda_mapping'],
                ['datasets/Unit conversion/conversion_new.txt', 'conversion']]
    for ds in datasets:
        ds_path, ds_name = ds[0], ds[1]
        with open(ds_path, 'r', encoding='utf8') as f:
            docs = f.readlines()
            dataset = []
            for line in docs:
                if len(line.strip()) > 0:
                    line = line.replace('[INST]', '').replace('\t', ' ')
                    instruction, output = line.strip().split('[/INST]')
                    dataset.append((instruction.replace(' ?', '?').replace('"', '').strip(), output.replace('"', '').strip()))
        df = pd.DataFrame(dataset, columns=['instruction', 'output'])
        for fold in range(5):
            if not os.path.exists(TRAIN_SET_PATH + '/' + f'split_{fold}/'):
                os.makedirs(TRAIN_SET_PATH + '/' + f'split_{fold}/')
            df.to_csv(TRAIN_SET_PATH + '/' + f'split_{fold}/' + ds_name + '.tsv', encoding='utf8', index=False, sep='\t')


def preprocess_ingredients():
    train_datasets = ['datasets/ingredients/ingredient_nutrient_value_training.txt']
    test_datasets = ['datasets/ingredients/ingredient_nutrient_value_test.txt']

    all_datasets = [train_datasets, test_datasets]
    for idx, dss in enumerate(all_datasets):
        for ds in dss:
            output_name = " ".join(ds.split('/')[1:]).replace(' ', '_').lower().split('.')[0]
            for fold in range(5):
                with open(ds, 'r', encoding='utf8') as f:
                    docs = f.readlines()
                    dataset = []
                    pair = []
                    for line in docs:
                        if len(line.strip()) > 0:
                            if '[INST]' in line:
                                inst = line.replace('[INST]', '').replace('[/INST]', '').replace('\t', ' ').strip()
                                pair.append(inst)
                            else:
                                output = line.strip()
                                pair.append(output)
                                if len(pair) == 2:
                                    dataset.append((pair[0], pair[1]))
                                    pair = []
                                else:
                                    print('Something is wrong', pair)
                                    break
                df = pd.DataFrame(dataset, columns=['instruction', 'output'])
                if idx == 0:
                    df.to_csv(TRAIN_SET_PATH + '/' + f'split_{fold}/' + output_name + '.tsv', encoding='utf8', index=False, sep='\t')
                elif idx == 1:
                    df.to_csv(TEST_SET_PATH + '/' + f'split_{fold}/' + output_name + '.tsv', encoding='utf8', index=False, sep='\t')


def preprocess_recipes():
    fsa_ingredients_seeds = ['120459', '228727', '398065', '537363', '638908']
    fsa_title_ingredients_seeds = ['150769', '359225', '476390', '755236', '812461']
    nutrient_ingredients_seeds = ['107473', '234053', '442417', '619176', '777572']
    nutrient_title_ingredients_seeds = ['206184', '384578', '512894', '638165', '767065']
    for fold, seeds in enumerate(zip(fsa_ingredients_seeds, fsa_title_ingredients_seeds, nutrient_ingredients_seeds, nutrient_title_ingredients_seeds)):
        train_datasets = [f'datasets/recipes/fsa lights/ingredients/{seeds[0]}/dataset_training_20_{seeds[0]}.txt',
                          f'datasets/recipes/fsa lights/title and ingredients/{seeds[1]}/dataset_training_20_{seeds[1]}.txt',
                          f'datasets/recipes/nutrient values/ingredients/{seeds[2]}/dataset_training_20_{seeds[2]}.txt',
                          f'datasets/recipes/nutrient values/title and ingredients/{seeds[3]}/dataset_training_20_{seeds[3]}.txt',
        ]

        test_datasets = [f'datasets/recipes/fsa lights/ingredients/{seeds[0]}/dataset_test_20_{seeds[0]}.txt',
                         f'datasets/recipes/fsa lights/title and ingredients/{seeds[1]}/dataset_test_20_{seeds[1]}.txt',
                         f'datasets/recipes/nutrient values/ingredients/{seeds[2]}/dataset_test_20_{seeds[2]}.txt',
                         f'datasets/recipes/nutrient values/title and ingredients/{seeds[3]}/dataset_test_20_{seeds[3]}.txt',
        ]

        all_datasets = [train_datasets, test_datasets]
        for idx, dss in enumerate(all_datasets):
            for ds in dss:
                output_name = " ".join(ds.split('/')[1:]).replace(' ', '_').lower().split('.')[0]
                with open(ds, 'r', encoding='utf8') as f:
                    docs = f.readlines()
                    dataset = []
                    pair = []
                    for line in docs:
                        if len(line.strip()) > 0:
                            if '[INST]' in line:
                                inst = line.replace('[INST]', '').replace('[/INST]', '').replace('\t', ' ').strip()
                                pair.append(inst)
                            else:
                                output = line.strip()
                                pair.append(output)
                                if len(pair) == 2:
                                    dataset.append((pair[0], pair[1]))
                                    pair = []
                                else:
                                    print('Something is wrong', pair)
                                    break
                df = pd.DataFrame(dataset, columns=['instruction', 'output'])
                if idx == 0:
                    df.to_csv(TRAIN_SET_PATH + '/' + f'split_{fold}/' + output_name + '.tsv', encoding='utf8', index=False, sep='\t')
                elif idx == 1:
                    df.to_csv(TEST_SET_PATH + '/' + f'split_{fold}/' + output_name + '.tsv', encoding='utf8', index=False, sep='\t')



if __name__ == '__main__':
    preprocess_ner()
    preprocess_nel_bootstrap()
    preprocess_USDA_FCD_and_conversion()
    preprocess_ingredients()
    preprocess_recipes()


















