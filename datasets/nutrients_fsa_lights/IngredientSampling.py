import json
import os
import random
from abc import ABC, abstractmethod
from fractions import Fraction

from src.Entities import Recipe
from src.Writer import Writer


class SequenceGenerator(ABC):
    start = '[INST]'
    end = '[/INST]'

    ingredient_question_prompts = [
        "Determine the nutrient values of",
        "Calculate the nutrient values of",
        "Find the nutrient values of",
        "Identify the nutritional values of",
        "Ascertain the nutrient values of",
        "Discover the nutrient values of",
        "Establish the nutrient values of",
        "Evaluate the nutrient values of",
        "Analyze the nutrient values of",
        "Compute the nutrient values of",
        "Assess the nutrient values of",
        "Review the nutrient values of",
        "Check the nutrient values of",
        "Verify the nutrient values of",
        "Gauge the nutrient values of",
        "Determine the nutritional content of",
        "Calculate the nutritional content of",
        "Find the nutritional content of",
        "Identify the nutritional content of",
        "Ascertain the nutritional content of",
        "Discover the nutritional content of",
        "Establish the nutritional content of",
        "Evaluate the nutritional content of",
        "Analyze the nutritional content of",
        "Compute the nutritional content of",
        "Assess the nutritional content of",
        "Review the nutritional content of",
        "Check the nutritional content of",
        "Verify the nutritional content of",
        "Gauge the nutritional content of",
        "Determine the nutritional values of",
        "Calculate the nutritional values of",
        "Find the nutritional values of",
        "Identify the nutrient content of",
        "Ascertain the nutrient content of",
        "Discover the nutrient content of",
        "Establish the nutrient content of",
        "Evaluate the nutrient content of",
        "Analyze the nutrient content of",
        "Compute the nutrient content of",
        "Assess the nutrient content of",
        "Review the nutrient content of",
        "Check the nutrient content of",
        "Verify the nutrient content of",
        "Gauge the nutrient content of",
        "Determine the nutrition values of",
        "Calculate the nutrition values of",
        "Find the nutrition values of",
        "Identify the nutrition values of",
        "Ascertain the nutrition values of",
        "Discover the nutrition values of",
        "Establish the nutrition values of",
        "Evaluate the nutrition values of",
        "Analyze the nutrition values of",
        "Compute the nutrition values of",
        "Assess the nutrition values of",
        "Review the nutrition values of",
        "Check the nutrition values of",
        "Verify the nutrition values of",
        "Gauge the nutrition values of",
        "Determine the nutritional profile of",
        "Calculate the nutritional profile of",
        "Find the nutritional profile of",
        "Identify the nutritional profile of",
        "Ascertain the nutritional profile of",
        "Discover the nutritional profile of",
        "Establish the nutritional profile of",
        "Evaluate the nutritional profile of",
        "Analyze the nutritional profile of",
        "Compute the nutritional profile of",
        "Assess the nutritional profile of",
        "Review the nutritional profile of",
        "Check the nutritional profile of",
        "Verify the nutritional profile of",
        "Gauge the nutritional profile of",
        "Determine the nutrient profile of",
        "Calculate the nutrient profile of",
        "Find the nutrient profile of",
        "Identify the nutrient profile of",
        "Ascertain the nutrient profile of",
        "Discover the nutrient profile of",
        "Establish the nutrient profile of",
        "Evaluate the nutrient profile of",
        "Analyze the nutrient profile of",
        "Compute the nutrient profile of",
        "Assess the nutrient profile of",
        "Review the nutrient profile of",
        "Check the nutrient profile of",
        "Verify the nutrient profile of",
        "Gauge the nutrient profile of",
        "Determine the nutrition facts of",
        "Calculate the nutrition facts of",
        "Find the nutrition facts of",
        "Identify the nutrition facts of",
        "Ascertain the nutrition facts of"
    ]

    ingredient_answer_prompts = [
        "The nutrient values are as follows:",
        "Listed below are the nutrient values:",
        "The following are the nutrient values:",
        "Nutrient values include:",
        "These are the nutrient values:",
        "The nutrient values comprise:",
        "Here are the nutrient values:",
        "The nutrient values include:",
        "Provided below are the nutrient values:",
        "The nutrient values given are:",
        "Nutrient values are:",
        "Displayed here are the nutrient values:",
        "The nutrient values presented are:",
        "The nutrient values listed are:",
        "Shown below are the nutrient values:",
        "The nutrient values consist of:",
        "The nutrient values detailed are:",
        "Outlined here are the nutrient values:",
        "The nutrient values mentioned are:",
        "Included here are the nutrient values:",
        "The nutrient values specified are:",
        "The nutrient values are detailed as:",
        "The nutrient values noted are:",
        "Nutrient values described are:",
        "The nutrient values found are:",
        "Nutrient values reported are:",
        "These nutrient values are:",
        "The nutrient values are summarized as:",
        "The nutrient values are outlined as:",
        "The nutrient values indicated are:",
        "The nutrient values stated are:",
        "The nutrient values enumerated are:",
        "The nutrient values referenced are:",
        "The nutrient values highlighted are:",
        "The nutrient values mentioned below are:",
        "Nutrient values stated below are:",
        "The nutrient values recorded are:",
        "Nutrient values outlined below are:",
        "The nutrient values expressed are:",
        "These are the detailed nutrient values:",
        "The following nutrient values are given:",
        "The nutrient values shown are:",
        "The nutrient values given below are:",
        "Outlined below are the nutrient values:",
        "These are the listed nutrient values:",
        "The nutrient values represented are:",
        "The nutrient values depicted are:",
        "Nutrient values indicated are:",
        "The following are the detailed nutrient values:",
        "These values of nutrients are:",
        "The nutrient values identified are:",
        "The nutrient values illustrated are:",
        "Nutrient values provided are:",
        "Nutrient values displayed are:",
        "Below are the nutrient values:",
        "Nutrient values are given as follows:",
        "The nutrient values tabulated are:",
        "Listed nutrient values are:",
        "The nutrient values presented below are:",
        "The nutrient values explained are:",
        "The nutrient values are recorded as:",
        "The nutrient values shown below are:",
        "Detailed below are the nutrient values:",
        "Nutrient values are summarized as follows:",
        "The nutrient values listed below are:",
        "The nutrient values featured are:",
        "The nutrient values demonstrated are:",
        "The nutrient values identified below are:",
        "Nutrient values noted below are:",
        "These listed nutrient values are:",
        "The nutrient values are depicted as:",
        "Herein are the nutrient values:",
        "The nutrient values detailed below are:",
        "Nutrient values are enumerated as follows:",
        "These nutrient values given are:",
        "The nutrient values noted above are:",
        "Nutrient values recorded below are:",
        "The nutrient values set out are:",
        "Nutrient values are illustrated as:",
        "The nutrient values are displayed as follows:",
        "Nutrient values highlighted below are:",
        "The nutrient values below are listed as:",
        "These detailed nutrient values are:",
        "The nutrient values explained below are:",
        "Nutrient values outlined are:",
        "The nutrient values outlined above are:",
        "These nutrient values are provided as:",
        "The nutrient values expressed below are:",
        "Nutrient values are presented as follows:",
        "The nutrient values shown above are:",
        "Listed above are the nutrient values:",
        "Nutrient values are identified as follows:",
        "The nutrient values shown are as follows:",
        "These values for nutrients are:",
        "Outlined above are the nutrient values:",
        "Presented below are the nutrient values:",
        "The nutrient values identified above are:",
        "These values of nutrients are listed as:",
        "Nutrient values given are as follows:",
        "The nutrient values presented above are:",
        "Nutrient values indicated below are:",
        "Nutrient values demonstrated below are:",
        "The nutrient values enumerated above are:",
        "These are the given nutrient values:",
        "The nutrient values shown here are:",
        "Nutrient values are demonstrated as:",
        "The following detailed nutrient values are:",
        "These nutrient values recorded are:",
        "The nutrient values provided below are:",
        "The nutrient values listed above are:",
        "Below are the detailed nutrient values:",
        "The nutrient values depicted below are:",
        "These nutrient values identified are:",
        "The nutrient values illustrated below are:",
        "Nutrient values described below are:",
        "The nutrient values specified below are:",
        "The following nutrient values are listed:",
        "Nutrient values recorded above are:",
        "The nutrient values outlined below are as follows:",
        "These detailed values of nutrients are:",
        "The nutrient values mentioned above are:",
        "Outlined nutrient values are:",
        "The following nutrient values are:",
        "The nutrient values provided above are:",
        "The nutrient values detailed above are:",
        "Nutrient values stated above are:",
        "The nutrient values shown below are as follows:",
        "These nutrient values detailed are:",
        "The nutrient values enumerated below are:",
        "Outlined values of nutrients are:",
        "These listed values of nutrients are:",
        "The nutrient values described above are:",
        "Nutrient values demonstrated above are:",
        "The nutrient values highlighted above are:",
        "These are the nutrient values presented:",
        "The nutrient values expressed above are:",
        "Nutrient values are shown as follows:",
        "These nutrient values specified are:",
        "The nutrient values explained above are:",
        "Nutrient values highlighted above are listed as:",
        "The nutrient values illustrated above are:",
        "The nutrient values demonstrated here are:",
        "The nutrient values depicted above are:",
        "These nutrient values enumerated are:",
        "The nutrient values highlighted here are:",
        "Nutrient values provided above are as follows:",
        "These nutrient values noted are:"
    ]

    def __init__(self):
        random.seed(42)

    @abstractmethod
    def generate_sequences(self, data_file: str) -> dict:
        raise NotImplementedError()

    @staticmethod
    def generate_random_number(start: int, end: int) -> int:
        number = random.randint(start, end)
        return number


class IngredientQuantityNutrientValueSequenceGenerator(SequenceGenerator):
    @staticmethod
    def convert(q: str):
        try:
            if '-' in q or 'to' in q:
                return None
            elif '/' in q:
                s = float(sum(Fraction(s) for s in q.split()))
                return s
            else:
                return float(q)
        except Exception:
            return None

    def generate_sequences(self, data_file: str) -> dict:
        questions = []
        questions_test = []
        answers = []
        answers_test = []

        all = []
        skipped = []
        candidates = []
        in_dictionary = []

        ingredient_count = dict()
        ingredient_nutritional = dict()

        with open(data_file, encoding="utf-8") as f:
            instances = json.load(f)
            for i, instance in enumerate(instances):
                recipe = Recipe(instance)

                for ingredient, quant, unit, fat, ngr, pro, sat, sod, sug in zip(recipe.ingredients,
                                                                                 recipe.ingredient_quantity,
                                                                                 recipe.ingredient_unit,
                                                                                 recipe.ingredient_fat,
                                                                                 recipe.ingredient_nrg,
                                                                                 recipe.ingredient_pro,
                                                                                 recipe.ingredient_sat,
                                                                                 recipe.ingredient_sod,
                                                                                 recipe.ingredient_sug):
                    all.append([i, ingredient, unit, quant])

                    # try to convert each unit to number
                    quantity = self.convert(quant)
                    if quantity is None:
                        skipped.append([i, ingredient, unit, quant])
                        continue

                    candidates.append([i, ingredient, unit, quant])

                    ingredient_unit = f'{unit} {ingredient}'
                    answer = (f'energy - {ngr:.2f}, fat - {fat:.2f}, protein - {pro:.2f}, '
                              f'salt - {sod:.2f}, saturates - {sat:.2f}, sugars - {sug:.2f}')
                    if ingredient_unit not in ingredient_count.keys():
                        in_dictionary.append([i, ingredient, unit, quant])
                        ingredient_count[ingredient_unit] = 1
                        quantities = dict()
                        quantities[quantity] = [quant, answer]
                        ingredient_nutritional[ingredient_unit] = quantities.copy()
                    else:
                        ingredient_count[ingredient_unit] = int(ingredient_count[ingredient_unit]) + 1
                        quantities = ingredient_nutritional[ingredient_unit].copy()
                        if quantity not in quantities.keys():
                            in_dictionary.append([i, ingredient, unit, quant])
                            quantities[quantity] = [quant, answer]
                            ingredient_nutritional[ingredient_unit] = quantities.copy()

        sample = []
        sample_keys = []
        for key in sorted(ingredient_count, key=ingredient_count.get, reverse=False):
            # sort the quantities dictionary
            counter = 0
            quantities = ingredient_nutritional[key].copy()
            for quantity in sorted(quantities, key=(lambda x: float(x)), reverse=False):
                sample.append([key, quantity, quantities[quantity]])
                sample_keys.append(f'{key} {quantities[quantity]}')
                counter += 1
                if counter == 6:
                    break

        sample_test = []
        sample_test_keys = []
        for key in sorted(ingredient_count, key=ingredient_count.get, reverse=False):
            # sort the quantities dictionary
            counter = 0
            quantities = ingredient_nutritional[key].copy()
            for quantity in sorted(quantities, key=(lambda x: float(x)), reverse=False):
                key_test = f'{key} {quantities[quantity]}'
                if key_test not in sample_keys:
                    sample_test.append([key, quantity, quantities[quantity]])
                    sample_test_keys.append(key_test)
                    counter += 1
                    if counter == 6:
                        break

        for key in sample_test_keys:
            assert key not in sample_keys

        for i, j, k in sample:
            questions.append(f'{self.start} '
                             f'{self.ingredient_question_prompts[self.generate_random_number(
                                 0, len(self.ingredient_question_prompts)-1)]} '
                             f'{k[0]} {i} {self.end}')
            answer = k[1]

            answers.append(
                f'{self.ingredient_answer_prompts[self.generate_random_number(0, len(self.ingredient_answer_prompts) - 1)]}'
                f' {answer}')

        for i, j, k in sample_test:
            questions_test.append(f'{self.start} '
                                  f'{self.ingredient_question_prompts[self.generate_random_number(
                                      0, len(self.ingredient_question_prompts) - 1)]} '
                                  f'{k[0]} {i} {self.end}')
            answer = k[1]

            answers_test.append(
                f'{self.ingredient_answer_prompts[self.generate_random_number(0, len(self.ingredient_answer_prompts) - 1)]}'
                f' {answer}')

        return {'questions': questions, 'answers': answers,
                'questions_test': questions_test, 'answers_test': answers_test}


if __name__ == '__main__':
    file = r'path\to\input\file'
    directory = r'path\to\output\directory'

    generator = IngredientQuantityNutrientValueSequenceGenerator()
    sequence = generator.generate_sequences(file)

    writer = Writer()
    writer.write(
        save_file=os.path.join(directory, f'ingredient_nutrient_value_training.txt'),
        questions=sequence['questions'],
        answers=sequence['answers'],
        append=False
    )

    writer.write(
        save_file=os.path.join(directory, f'ingredient_nutrient_value_test.txt'),
        questions=sequence['questions_test'],
        answers=sequence['answers_test'],
        append=False
    )
