# Code for experiments conducted in the paper 'FoodyLLM: A FAIR-aligned specialized large language model for food and nutrition analysis' #

## Also checkout the trained model available on HuggingFace: https://huggingface.co/Matej/FoodyLLM

## The model is based on Meta-Llama-3-8B-Instruct, which was fine-tuned (using LoRA) for food and nutrition analysis. 

More specifically, it can conduct the following tasks:
- Assessing recipe nutritional profiles
- Classifying recipes by traffic light nutrition labels (see https://www.food.gov.uk/safety-hygiene/check-the-label for details on the labeling)
- Extract food named entities from text (Food NER)
- Link the food entities to three distinct ontologies, Hansard taxonomy, FoodOn and SNOMED-CT (Food NEL)


## Datasets ##

TODO: All datasets will be added soon!

## Installation, documentation ##

Published results were produced in Python 3.12 programming environment on AlmaLinux 8.10 (Cerulean Leopard) operating system. Instructions for installation assume the usage of PyPI package manager and availability of CUDA (we use version 12.8).<br/>

Clone the project from the repository.<br/>

Install dependencies if needed: pip install -r requirements.txt

### To reproduce the results published in the paper, run the code in the command line using following commands: ###

Data preprocessing - split  datasets into 5 train and test folds and make it appropriate to feed into language model:<br/>

```
python preprocess.py
```

The script creates two folders (by default 'train_sets_ner_nel' and 'test_sets_ner_nel') containing the train and test datasets for 5 folds. <br/>

Train and test 5 language models (one for each fold) on five folds:<br/>

```
python train_and_test.py
```

The script trains and saves the models and the models' outputs on the test sets in the results folder by default.<br/>

To apply the model on the new data, run the example apply script:<br/>

```
python apply.py
```

The script by default downloads the trained FoodSEM model from the Huggingface library.<br/>

Additionally, we offer script that we used for our baseline zero-shot and few-shot experiments:<br/>

```
python test_incontext.py
```

## To reproduce results for our benchmark study titled 'A Domain-Targeted Question-Answering Dataset for Food and Nutrition Applications'

Use the same data preprocessing script as above to preprocess data and generate splits:<br/>

```
python preprocess.py
```

Scripts for testing several LLMs in zero-shot and few-shot settings are in the benchmark folder. You can run them with: <br/>

```
python name_of_the_script.py
```

Note that to reproduce Gemini-2.5-flash results you need to obtain Gemini API key.<br/>







