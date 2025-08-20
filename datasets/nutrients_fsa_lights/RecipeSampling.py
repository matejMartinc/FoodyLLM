import json
import os
import random
from abc import ABC, abstractmethod

from src.Entities import Recipe
from src.Writer import Writer


class SequenceGenerator(ABC):
    start = '[INST]'
    end = '[/INST]'

    def __init__(self, random_seed: int, max_count: int, max_jaccard: float):
        random.seed(random_seed)
        self.random_seed = random_seed
        self.max_count = max_count
        self.max_jaccard = max_jaccard

    @abstractmethod
    def sample(self, count: dict, recipes: dict, used_recipe_ids: list,
               used_recipe_titles: list, used_recipe_ingredients: list) -> [list, list, list, list, list, list]:
        raise NotImplementedError()

    @abstractmethod
    def generate(self, sample: list, ingredients: list) -> dict:
        raise NotImplementedError()

    def generate_sequences(self, data_file: str) -> dict:
        [count, recipes] = self.read_file(data_file)
        [sample, ingredients, recipe_ids, recipe_titles, recipe_ingredient_strings] = self.sample(
            count, recipes, [], [], [])
        [sample_test, ingredients_test, _, _, _] = self.sample(count, recipes, recipe_ids, recipe_titles, recipe_ingredient_strings)
        prompts = self.generate(sample, ingredients)
        prompts_test = self.generate(sample_test, ingredients_test)
        return prompts, prompts_test

    @staticmethod
    def jaccard_similarity(list_a: list, list_b: list) -> float:
        if list_a == list_b:
            return 1.0
        s1 = set(list_a)
        s2 = set(list_b)
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))

    @staticmethod
    def generate_random_number(start: int, end: int) -> int:
        number = random.randint(start, end)
        return number

    @staticmethod
    def generate_shuffled_indexes(count: int):
        indexes = list(range(count))
        random.shuffle(indexes)
        return indexes

    @staticmethod
    def read_file(data_file: str) -> [dict, dict]:
        count = dict()
        recipes = dict()

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
                    ingredient_unit = f'{unit} {ingredient}'

                    if ingredient_unit not in count.keys():
                        count[ingredient_unit] = 1
                        recipes_ = [recipe]
                        recipes[ingredient_unit] = recipes_.copy()
                    else:
                        count[ingredient_unit] = int(count[ingredient_unit]) + 1
                        recipes_ = recipes[ingredient_unit].copy()
                        recipes_.append(recipe)
                        recipes[ingredient_unit] = recipes_.copy()

        return [count, recipes]


class RecipeSequenceGenerator(SequenceGenerator, ABC):
    def sample(self, count: dict, recipes: dict,
               used_recipe_ids: list, used_recipe_titles: list, used_recipe_ingredients: list) -> [list, list, list, list, list, list]:
        recipe_id_in_sample = []
        recipe_title_in_sample = []
        recipe_ingredients_in_sample = []
        ingredient_count = dict()

        i = 0
        sample = []
        for key in sorted(count, key=count.get, reverse=False):
            # sort the quantities dictionary
            i += 1
            recipes_ = recipes[key]
            indexes = self.generate_shuffled_indexes(count=len(recipes_))
            recipes_shuffled = [recipes_[i] for i in indexes]

            counter = 0
            for recipe_temp in recipes_shuffled:
                ingredient_set_temp = []
                for ingredient in sorted(recipe_temp.ingredients):
                    ingredient_set_temp.append(ingredient.lower())

                if recipe_temp.id_ in recipe_id_in_sample:
                    continue
                if recipe_temp.id_ in used_recipe_ids:
                    continue

                if recipe_temp.title.lower() in recipe_title_in_sample:
                    continue
                if recipe_temp.title.lower() in used_recipe_titles:
                    continue

                if key in ingredient_count.keys() and int(ingredient_count[key]) >= self.max_count:
                    continue

                # find max jaccard similarity with currently sampled set
                max_ = 0.0
                for sampled_set in recipe_ingredients_in_sample:
                    jaccard_similarity = self.jaccard_similarity(ingredient_set_temp, sampled_set)
                    if jaccard_similarity > max_:
                        max_ = jaccard_similarity
                        if max_ >= self.max_jaccard:
                            break

                if max_ >= self.max_jaccard:
                    continue

                # find max jaccard similarity with previously sampled set (training set)
                max_ = 0.0
                for sampled_set in used_recipe_ingredients:
                    jaccard_similarity = self.jaccard_similarity(ingredient_set_temp, sampled_set)
                    if jaccard_similarity > max_:
                        max_ = jaccard_similarity
                        if max_ >= self.max_jaccard:
                            break

                if max_ >= self.max_jaccard:
                    continue

                sample.append([recipe_temp.id_, recipe_temp, recipe_temp.title, '; '.join(sorted(ingredient_set_temp))])
                recipe_id_in_sample.append(recipe_temp.id_)
                recipe_title_in_sample.append(recipe_temp.title.lower())
                recipe_ingredients_in_sample.append(ingredient_set_temp)

                # update sample_ingredient_count dictionary after taking the sample
                for ing, un in zip(recipe_temp.ingredients, recipe_temp.ingredient_unit):
                    ing_un = f'{un} {ing}'
                    if ing_un not in ingredient_count.keys():
                        ingredient_count[ing_un] = 1
                    else:
                        ingredient_count[ing_un] = int(ingredient_count[ing_un]) + 1

                counter += 1
                if counter == self.max_count:
                    break

        return [sample, [[k, ingredient_count[k]] for k in sorted(
            ingredient_count, key=ingredient_count.get, reverse=True)],
                recipe_id_in_sample, recipe_title_in_sample, recipe_ingredients_in_sample]


class RecipeIngrNutritionSequenceGenerator(RecipeSequenceGenerator):
    recipe_question_prompts = [
        "Determine the nutritional values per 100 g in a recipe that includes these ingredients:",
        "Calculate the nutrient content per 100 g in a recipe with the following ingredients:",
        "Find the nutrient values per 100 grams in a recipe containing these ingredients:",
        "Identify the nutritional content per 100 grams for a recipe with the following ingredients:",
        "Ascertain the nutrient values per 100 g in a recipe using these ingredients:",
        "Discover the nutrient values per 100 g in a recipe that uses the following ingredients:",
        "Establish the nutrient values per 100 grams in a recipe that contains these ingredients:",
        "Evaluate the nutrient values per 100 grams in a recipe made with these ingredients:",
        "Analyze the nutrient values per 100 g in a recipe with the following ingredients:",
        "Compute the nutrient values per 100 g in a recipe that includes these ingredients:",
        "Assess the nutrient values per 100 grams in a recipe containing the following ingredients:",
        "Review the nutrient values per 100 grams in a recipe that comprises these ingredients:",
        "Check the nutrient values per 100 g in a recipe that has these ingredients:",
        "Verify the nutrient values per 100 g in a recipe prepared with these ingredients:",
        "Gauge the nutrient values per 100 grams in a recipe that incorporates the following ingredients:",
        "Determine the nutrient composition per 100 grams in a recipe containing these ingredients:",
        "Calculate the nutrient breakdown per 100 g in a recipe with these ingredients:",
        "Find the nutritional breakdown per 100 g in a recipe that uses the following ingredients:",
        "Identify the nutritional profile per 100 grams in a recipe containing these ingredients:",
        "Ascertain the nutrient profile per 100 grams in a recipe with the following ingredients:",
        "Discover the nutrient composition per 100 g in a recipe that includes these ingredients:",
        "Establish the nutritional content per 100 g in a recipe with these ingredients:",
        "Evaluate the nutrient composition per 100 grams in a recipe using the following ingredients:",
        "Analyze the nutritional profile per 100 grams in a recipe made with these ingredients:",
        "Compute the nutritional values per 100 g in a recipe containing these ingredients:",
        "Assess the nutritional breakdown per 100 g in a recipe with the following ingredients:",
        "Review the nutritional profile per 100 grams in a recipe that comprises these ingredients:",
        "Check the nutrient composition per 100 grams in a recipe that includes these ingredients:",
        "Verify the nutrient breakdown per 100 g in a recipe that contains the following ingredients:",
        "Gauge the nutritional content per 100 g in a recipe made with these ingredients:",
        "Determine the nutrient values per 100 grams in a recipe using the following ingredients:",
        "Calculate the nutrient profile per 100 grams in a recipe with these ingredients:",
        "Find the nutrient breakdown per 100 g in a recipe containing the following ingredients:",
        "Identify the nutritional values per 100 g in a recipe prepared with these ingredients:",
        "Ascertain the nutritional breakdown per 100 grams in a recipe that includes these ingredients:",
        "Discover the nutritional profile per 100 grams in a recipe with these ingredients:",
        "Establish the nutrient profile per 100 g in a recipe containing these ingredients:",
        "Evaluate the nutrient content per 100 g in a recipe that uses the following ingredients:",
        "Analyze the nutritional composition per 100 grams in a recipe made with these ingredients:",
        "Compute the nutrient profile per 100 grams in a recipe that includes these ingredients:",
        "Assess the nutrient composition per 100 g in a recipe using the following ingredients:",
        "Review the nutrient breakdown per 100 g in a recipe containing these ingredients:",
        "Check the nutritional profile per 100 g in a recipe with the following ingredients:",
        "Verify the nutritional breakdown per 100 grams in a recipe that comprises these ingredients:",
        "Gauge the nutrient values per 100 grams in a recipe that has these ingredients:",
        "Determine the nutritional breakdown per 100 g in a recipe made with these ingredients:",
        "Calculate the nutrient composition per 100 g in a recipe using the following ingredients:",
        "Find the nutritional content per 100 grams in a recipe with the following ingredients:",
        "Identify the nutrient breakdown per 100 grams in a recipe that includes these ingredients:",
        "Ascertain the nutrient values per 100 g in a recipe that comprises these ingredients:",
        "Discover the nutrient content per 100 g in a recipe that has these ingredients:",
        "Establish the nutritional values per 100 grams in a recipe with the following ingredients:",
        "Evaluate the nutrient breakdown per 100 grams in a recipe that contains these ingredients:",
        "Analyze the nutrient profile per 100 g in a recipe using the following ingredients:",
        "Compute the nutritional profile per 100 g in a recipe made with these ingredients:",
        "Assess the nutrient values per 100 grams in a recipe that includes the following ingredients:",
        "Review the nutritional values per 100 grams in a recipe prepared with these ingredients:",
        "Check the nutrient profile per 100 g in a recipe using these ingredients:",
        "Verify the nutrient values per 100 g in a recipe containing the following ingredients:",
        "Gauge the nutritional breakdown per 100 grams in a recipe with the following ingredients:",
        "Determine the nutritional profile per 100 grams in a recipe that uses these ingredients:",
        "Calculate the nutritional composition per 100 g in a recipe containing these ingredients:",
        "Find the nutrient values per 100 g in a recipe with these ingredients:",
        "Identify the nutrient content per 100 grams in a recipe that includes the following ingredients:",
        "Ascertain the nutrient values per 100 grams in a recipe that has these ingredients:",
        "Discover the nutrient profile per 100 g in a recipe containing these ingredients:",
        "Establish the nutrient breakdown per 100 g in a recipe with the following ingredients:",
        "Evaluate the nutritional profile per 100 grams in a recipe using these ingredients:",
        "Analyze the nutrient values per 100 grams in a recipe made with these ingredients:",
        "Compute the nutrient breakdown per 100 g in a recipe that includes these ingredients:",
        "Assess the nutritional profile per 100 g in a recipe containing these ingredients:",
        "Review the nutrient values per 100 grams in a recipe with these ingredients:",
        "Check the nutritional content per 100 grams in a recipe that uses the following ingredients:",
        "Verify the nutrient profile per 100 g in a recipe containing these ingredients:",
        "Gauge the nutrient content per 100 g in a recipe that comprises these ingredients:",
        "Determine the nutrient profile per 100 grams in a recipe with these ingredients:",
        "Calculate the nutrient values per 100 grams in a recipe that includes these ingredients:",
        "Find the nutrient profile per 100 g in a recipe that contains these ingredients:",
        "Identify the nutrient composition per 100 g in a recipe using the following ingredients:",
        "Ascertain the nutrient values per 100 grams in a recipe that includes these ingredients:",
        "Discover the nutritional content per 100 grams in a recipe with these ingredients:",
        "Establish the nutrient values per 100 g in a recipe containing the following ingredients:",
        "Evaluate the nutritional breakdown per 100 g in a recipe that includes these ingredients:",
        "Analyze the nutrient composition per 100 grams in a recipe using these ingredients:",
        "Compute the nutrient values per 100 grams in a recipe with the following ingredients:",
        "Assess the nutritional content per 100 g in a recipe containing these ingredients:",
        "Review the nutrient values per 100 g in a recipe that comprises these ingredients:",
        "Check the nutrient values per 100 grams in a recipe that includes the following ingredients:",
        "Verify the nutritional profile per 100 grams in a recipe made with these ingredients:",
        "Gauge the nutritional values per 100 g in a recipe prepared with these ingredients:",
        "Determine the nutrient content per 100 g in a recipe that has these ingredients:",
        "Calculate the nutritional profile per 100 grams in a recipe that contains these ingredients:",
        "Find the nutrient breakdown per 100 grams in a recipe using the following ingredients:",
        "Identify the nutrient values per 100 g in a recipe that includes these ingredients:",
        "Ascertain the nutritional profile per 100 g in a recipe with the following ingredients:",
        "Discover the nutrient breakdown per 100 grams in a recipe that comprises these ingredients:",
        "Establish the nutritional content per 100 grams in a recipe that contains these ingredients:",
        "Evaluate the nutrient profile per 100 g in a recipe made with these ingredients:",
        "Analyze the nutritional values per 100 g in a recipe that has these ingredients:",
        "Compute the nutrient breakdown per 100 grams in a recipe using these ingredients:",
        "Assess the nutrient profile per 100 grams in a recipe containing the following ingredients:",
        "Review the nutritional composition per 100 g in a recipe with these ingredients:",
        "Check the nutrient profile per 100 g in a recipe that includes these ingredients:",
        "Verify the nutrient composition per 100 grams in a recipe that uses these ingredients:",
        "Gauge the nutrient breakdown per 100 grams in a recipe that includes the following ingredients:",
        "Determine the nutritional values per 100 g in a recipe that contains these ingredients:",
        "Calculate the nutrient content per 100 g in a recipe with the following ingredients:",
        "Find the nutrient profile per 100 grams in a recipe that includes these ingredients:",
        "Identify the nutritional composition per 100 grams in a recipe with these ingredients:",
        "Ascertain the nutritional values per 100 g in a recipe that contains these ingredients:",
        "Discover the nutrient values per 100 g in a recipe prepared with these ingredients:",
        "Establish the nutrient breakdown per 100 grams in a recipe using the following ingredients:",
        "Evaluate the nutritional profile per 100 grams in a recipe that includes these ingredients:",
        "Analyze the nutrient composition per 100 g in a recipe containing these ingredients:",
        "Compute the nutritional values per 100 g in a recipe with these ingredients:",
        "Assess the nutrient breakdown per 100 grams in a recipe that includes the following ingredients:",
        "Review the nutrient values per 100 grams in a recipe using these ingredients:",
        "Check the nutritional values per 100 g in a recipe that comprises these ingredients:",
        "Verify the nutrient content per 100 g in a recipe that has these ingredients:",
        "Gauge the nutrient composition per 100 grams in a recipe with the following ingredients:",
        "Determine the nutrient values per 100 grams in a recipe that uses these ingredients:",
        "Calculate the nutritional values per 100 g in a recipe containing the following ingredients:",
        "Find the nutritional profile per 100 g in a recipe with the following ingredients:",
        "Identify the nutrient breakdown per 100 grams in a recipe prepared with these ingredients:",
        "Ascertain the nutrient values per 100 grams in a recipe using these ingredients:",
        "Discover the nutritional values per 100 g in a recipe that includes these ingredients:",
        "Establish the nutrient composition per 100 g in a recipe that comprises these ingredients:",
        "Evaluate the nutrient values per 100 grams in a recipe that has these ingredients:",
        "Analyze the nutritional breakdown per 100 grams in a recipe with the following ingredients:",
        "Compute the nutrient values per 100 g in a recipe that contains these ingredients:",
        "Assess the nutrient profile per 100 g in a recipe that includes these ingredients:",
        "Review the nutrient composition per 100 grams in a recipe using the following ingredients:",
        "Check the nutrient values per 100 grams in a recipe that comprises these ingredients:",
        "Verify the nutritional profile per 100 g in a recipe with these ingredients:",
        "Gauge the nutrient profile per 100 g in a recipe using these ingredients:",
        "Determine the nutrient values per 100 grams in a recipe that has these ingredients:",
        "Calculate the nutrient profile per 100 grams in a recipe containing the following ingredients:",
        "Find the nutritional breakdown per 100 g in a recipe that includes these ingredients:",
        "Identify the nutritional profile per 100 g in a recipe that uses these ingredients:",
        "Ascertain the nutrient values per 100 grams in a recipe with the following ingredients:",
        "Discover the nutrient values per 100 grams in a recipe that includes these ingredients:",
        "Establish the nutrient values per 100 g in a recipe using these ingredients:",
        "Evaluate the nutritional profile per 100 g in a recipe that has these ingredients:",
        "Analyze the nutrient values per 100 grams in a recipe with the following ingredients:",
        "Compute the nutrient values per 100 grams in a recipe that comprises these ingredients:",
        "Assess the nutrient profile per 100 g in a recipe that includes the following ingredients:",
        "Review the nutrient profile per 100 g in a recipe containing these ingredients:",
        "Check the nutrient values per 100 grams in a recipe using these ingredients:",
        "Verify the nutritional breakdown per 100 grams in a recipe that has these ingredients:",
        "Gauge the nutrient values per 100 g in a recipe containing the following ingredients:",
        "Determine the nutritional values per 100 g in a recipe that comprises these ingredients:",
        "Calculate the nutrient values per 100 grams in a recipe with these ingredients:",
        "Find the nutrient profile per 100 grams in a recipe using these ingredients:",
        "Identify the nutritional values per 100 g in a recipe that includes the following ingredients:",
        "Ascertain the nutrient breakdown per 100 g in a recipe with these ingredients:",
        "Discover the nutrient profile per 100 grams in a recipe that contains the following ingredients:",
        "Establish the nutritional breakdown per 100 grams in a recipe that includes these ingredients:",
        "Evaluate the nutrient composition per 100 g in a recipe that has these ingredients:",
        "Analyze the nutrient profile per 100 g in a recipe with these ingredients:",
        "Compute the nutritional content per 100 grams in a recipe that includes the following ingredients:",
        "Assess the nutrient values per 100 grams in a recipe with these ingredients:",
        "Review the nutritional values per 100 g in a recipe containing these ingredients:",
        "Check the nutrient breakdown per 100 g in a recipe that includes these ingredients:",
        "Verify the nutritional content per 100 grams in a recipe that comprises these ingredients:",
        "Gauge the nutrient values per 100 grams in a recipe prepared with the following ingredients:",
        "Determine the nutrient values per 100 g in a recipe that uses the following ingredients:",
        "Calculate the nutritional values per 100 g in a recipe with these ingredients:",
        "Find the nutrient composition per 100 grams in a recipe containing these ingredients:",
        "Identify the nutrient values per 100 grams in a recipe that includes the following ingredients:",
        "Ascertain the nutrient values per 100 g in a recipe with these ingredients:",
        "Discover the nutrient breakdown per 100 g in a recipe prepared with these ingredients:",
        "Establish the nutrient profile per 100 grams in a recipe that includes the following ingredients:",
        "Evaluate the nutrient values per 100 grams in a recipe with these ingredients:",
        "Analyze the nutrient values per 100 g in a recipe that uses these ingredients:",
        "Compute the nutritional profile per 100 g in a recipe that comprises these ingredients:",
        "Assess the nutrient values per 100 grams in a recipe with the following ingredients:",
        "Review the nutrient breakdown per 100 grams in a recipe containing these ingredients:",
        "Check the nutritional values per 100 g in a recipe that includes these ingredients:",
        "Verify the nutrient breakdown per 100 g in a recipe prepared with these ingredients:",
        "Gauge the nutrient values per 100 g in a recipe that has the following ingredients:"
    ]
    recipe_answer_prompts = [
        "The nutritional values per 100 g are:",
        "Nutrient content per 100 g:",
        "Per 100 g, the nutrient values are:",
        "Per 100 grams, the nutrient values are:",
        "The nutrient content for every 100 g is:",
        "Nutritional values for each 100 g:",
        "Each 100 g contains the following nutrient values:",
        "Nutrient breakdown per 100 g:",
        "The nutrient profile per 100 g is:",
        "Nutrient values in 100 g:",
        "Nutritional information per 100 g:",
        "Values per 100 g of nutrients:",
        "Nutrient facts per 100 g:",
        "For every 100 g, the nutrients are:",
        "Each 100 grams has these nutrient values:",
        "Nutrient content in 100 g:",
        "Nutrient values in each 100 g:",
        "Nutritional breakdown per 100 g:",
        "Nutrient values per 100 grams:",
        "The nutritional profile per 100 g is:",
        "Nutrient details per 100 g:",
        "Nutritional data per 100 g:",
        "Nutrient information per 100 g:",
        "The nutrients per 100 g are:",
        "Each 100 g provides these nutrient values:",
        "The nutrient values in 100 g are:",
        "In 100 g, the nutrient values are:",
        "For 100 g, the nutrient values are:",
        "Nutritional content for every 100 g:",
        "Nutrient breakdown in 100 g:",
        "Nutrient values per 100 g listed:",
        "Every 100 g has these nutrient values:",
        "Per 100 g nutrient breakdown:",
        "Nutrient details in 100 g:",
        "Nutrient profile for each 100 g:",
        "The nutrient values found in 100 g are:",
        "In each 100 g, the nutrient values are:",
        "Nutrient facts in 100 g:",
        "Nutritional content per 100 grams:",
        "Nutritional breakdown in 100 g:",
        "Values of nutrients per 100 g:",
        "The nutrient values for 100 g are:",
        "Nutritional profile per 100 grams:",
        "Nutrient values in every 100 g:",
        "Nutrient information in 100 g:",
        "Per 100 g, the nutrients are:",
        "Nutritional details per 100 g:",
        "Nutrient values in each 100 grams:",
        "The nutrient values per 100 grams are:",
        "Every 100 grams contains the following nutrient values:",
        "Nutrient content per 100 grams:",
        "Nutrient values in per 100 g:",
        "The nutrient content for each 100 g is:",
        "For every 100 g, the nutrient values are:",
        "Each 100 g contains these nutrient values:",
        "Nutrient values in 100 grams:",
        "Nutrient profile in 100 g:",
        "Nutritional values in 100 g:",
        "Per 100 g, the nutritional values are:",
        "The nutrient breakdown per 100 g is:",
        "Nutritional facts per 100 g:",
        "Nutrient details per 100 grams:",
        "Nutrient content in each 100 g:",
        "For 100 grams, the nutrient values are:",
        "Nutritional profile in 100 grams:",
        "Nutrient values listed per 100 g:",
        "Each 100 grams provides these nutrient values:",
        "In every 100 g, the nutrient values are:",
        "Nutritional values per 100 grams:",
        "The nutrient values in each 100 g are:",
        "Nutrient values in every 100 grams:",
        "Values of nutrients in 100 g:",
        "Nutritional information in 100 g:",
        "Each 100 g has these nutrient values:",
        "Nutrient breakdown per 100 grams:",
        "Nutrient facts per 100 grams:",
        "Nutritional breakdown for every 100 g:",
        "Nutrient profile per 100 g listed:",
        "The nutrient content per 100 grams is:",
        "Nutritional values found in 100 g:",
        "For every 100 grams, the nutrient values are:",
        "Nutritional values in each 100 g:",
        "Nutrient details in 100 grams:",
        "In 100 grams, the nutrient values are:",
        "Each 100 g provides the following nutrient values:",
        "Nutrient values per 100 g given:",
        "Nutritional values listed per 100 g:",
        "The nutrients in 100 g are:",
        "Nutrient breakdown for each 100 g:",
        "Nutrient profile per 100 grams:",
        "Nutrient values per 100 g presented:",
        "The nutrient values for each 100 g are:",
        "Nutritional details in 100 g:",
        "Every 100 g has the following nutrient values:",
        "Nutrient content for each 100 g:",
        "Per 100 grams nutrient breakdown:",
        "Nutrient profile in each 100 g:",
        "Nutritional facts in 100 g:",
        "Values of nutrients per 100 grams:",
        "The nutrient values in 100 grams are:",
        "Nutrient details for every 100 g:",
        "Nutrient values for 100 g listed:",
        "Each 100 g contains the following nutrient values:",
        "Nutrient information in each 100 g:",
        "The nutrient values per 100 g listed are:",
        "Nutrient values per 100 grams given:",
        "For each 100 g, the nutrient values are:",
        "The nutrient content per 100 g listed is:",
        "Nutritional content in 100 g:",
        "Nutritional values per 100 g provided:",
        "Nutrient values per 100 g detailed:",
        "Nutrient values listed in 100 g:",
        "The nutrients for each 100 g are:",
        "Nutrient values per 100 g found:",
        "Per 100 grams, the nutritional values are:",
        "Nutritional breakdown for 100 grams:",
        "Nutrient values per 100 g shown:",
        "Nutritional values for each 100 grams:",
        "Each 100 grams has these nutrient values:",
        "Nutrient profile for every 100 g:",
        "Nutrient values per 100 grams listed:",
        "Nutritional values for 100 g are:",
        "Nutrient values for each 100 grams:",
        "Nutrient content per 100 g detailed:",
        "The nutritional content per 100 g is:",
        "Nutrient values in each 100 grams are:",
        "Per 100 grams the nutrient values are:",
        "Nutrient values per 100 g provided:",
        "Nutritional values given per 100 g:",
        "Nutrient values for 100 grams:",
        "Nutrient profile detailed per 100 g:",
        "The nutrient values in every 100 g are:",
        "Nutrient values per 100 g highlighted:",
        "Nutrient values detailed for 100 g:",
        "Nutritional values in every 100 g:",
        "Nutrient values detailed per 100 grams:",
        "Each 100 g has these nutritional values:",
        "Nutrient values shown for 100 g:",
        "Nutrient values highlighted per 100 g:",
        "The nutrient values detailed for 100 g are:",
        "Nutrient values highlighted for 100 grams:",
        "The nutrient values provided per 100 g are:",
        "Nutrient values provided per 100 grams:",
        "Nutrient values per 100 g listed above:",
        "The nutrient values per 100 grams given are:",
        "Nutrient values per 100 grams shown:",
        "Each 100 grams provides the following nutrient values:",
        "The nutrient values highlighted per 100 g are:"
    ]

    def generate(self, sample: list, ingredients: list) -> dict:
        questions = []
        answers = []

        for i, recipe, _, _ in sample:
            question = ''
            for ingredient, quantity, unit in zip(recipe.ingredients, recipe.ingredient_quantity,
                                                  recipe.ingredient_unit):
                question += f'{quantity} {unit} {ingredient}, '
            question = question[:-2]

            answer = (f'energy - {recipe.energy_value:.2f}, fat - {recipe.fat_value:.2f}, '
                      f'protein - {recipe.protein_value:.2f}, salt - {recipe.salt_value:.2f}, '
                      f'saturates - {recipe.saturates_value:.2f}, sugars - {recipe.sugars_value:.2f}')

            questions.append(
                f'{self.start} '
                f'{self.recipe_question_prompts[
                    self.generate_random_number(0, len(self.recipe_question_prompts) - 1)]}'
                f' {question.strip()} {self.end}')

            answers.append(
                f'{self.recipe_answer_prompts[
                    self.generate_random_number(0, len(self.recipe_answer_prompts) - 1)]}'
                f' {answer}')

        return {'questions': questions, 'answers': answers}


class RecipeTitlIngrNutritionSequenceGenerator(RecipeSequenceGenerator):
    recipe_question_prompts = [
        "Determine the nutritional values per 100 g in a recipe that includes these ingredients:",
        "Calculate the nutrient content per 100 g in a recipe with the following ingredients:",
        "Find the nutrient values per 100 grams in a recipe containing these ingredients:",
        "Identify the nutritional content per 100 grams for a recipe with the following ingredients:",
        "Ascertain the nutrient values per 100 g in a recipe using these ingredients:",
        "Discover the nutrient values per 100 g in a recipe that uses the following ingredients:",
        "Establish the nutrient values per 100 grams in a recipe that contains these ingredients:",
        "Evaluate the nutrient values per 100 grams in a recipe made with these ingredients:",
        "Analyze the nutrient values per 100 g in a recipe with the following ingredients:",
        "Compute the nutrient values per 100 g in a recipe that includes these ingredients:",
        "Assess the nutrient values per 100 grams in a recipe containing the following ingredients:",
        "Review the nutrient values per 100 grams in a recipe that comprises these ingredients:",
        "Check the nutrient values per 100 g in a recipe that has these ingredients:",
        "Verify the nutrient values per 100 g in a recipe prepared with these ingredients:",
        "Gauge the nutrient values per 100 grams in a recipe that incorporates the following ingredients:",
        "Determine the nutrient composition per 100 grams in a recipe containing these ingredients:",
        "Calculate the nutrient breakdown per 100 g in a recipe with these ingredients:",
        "Find the nutritional breakdown per 100 g in a recipe that uses the following ingredients:",
        "Identify the nutritional profile per 100 grams in a recipe containing these ingredients:",
        "Ascertain the nutrient profile per 100 grams in a recipe with the following ingredients:",
        "Discover the nutrient composition per 100 g in a recipe that includes these ingredients:",
        "Establish the nutritional content per 100 g in a recipe with these ingredients:",
        "Evaluate the nutrient composition per 100 grams in a recipe using the following ingredients:",
        "Analyze the nutritional profile per 100 grams in a recipe made with these ingredients:",
        "Compute the nutritional values per 100 g in a recipe containing these ingredients:",
        "Assess the nutritional breakdown per 100 g in a recipe with the following ingredients:",
        "Review the nutritional profile per 100 grams in a recipe that comprises these ingredients:",
        "Check the nutrient composition per 100 grams in a recipe that includes these ingredients:",
        "Verify the nutrient breakdown per 100 g in a recipe that contains the following ingredients:",
        "Gauge the nutritional content per 100 g in a recipe made with these ingredients:",
        "Determine the nutrient values per 100 grams in a recipe using the following ingredients:",
        "Calculate the nutrient profile per 100 grams in a recipe with these ingredients:",
        "Find the nutrient breakdown per 100 g in a recipe containing the following ingredients:",
        "Identify the nutritional values per 100 g in a recipe prepared with these ingredients:",
        "Ascertain the nutritional breakdown per 100 grams in a recipe that includes these ingredients:",
        "Discover the nutritional profile per 100 grams in a recipe with these ingredients:",
        "Establish the nutrient profile per 100 g in a recipe containing these ingredients:",
        "Evaluate the nutrient content per 100 g in a recipe that uses the following ingredients:",
        "Analyze the nutritional composition per 100 grams in a recipe made with these ingredients:",
        "Compute the nutrient profile per 100 grams in a recipe that includes these ingredients:",
        "Assess the nutrient composition per 100 g in a recipe using the following ingredients:",
        "Review the nutrient breakdown per 100 g in a recipe containing these ingredients:",
        "Check the nutritional profile per 100 g in a recipe with the following ingredients:",
        "Verify the nutritional breakdown per 100 grams in a recipe that comprises these ingredients:",
        "Gauge the nutrient values per 100 grams in a recipe that has these ingredients:",
        "Determine the nutritional breakdown per 100 g in a recipe made with these ingredients:",
        "Calculate the nutrient composition per 100 g in a recipe using the following ingredients:",
        "Find the nutritional content per 100 grams in a recipe with the following ingredients:",
        "Identify the nutrient breakdown per 100 grams in a recipe that includes these ingredients:",
        "Ascertain the nutrient values per 100 g in a recipe that comprises these ingredients:",
        "Discover the nutrient content per 100 g in a recipe that has these ingredients:",
        "Establish the nutritional values per 100 grams in a recipe with the following ingredients:",
        "Evaluate the nutrient breakdown per 100 grams in a recipe that contains these ingredients:",
        "Analyze the nutrient profile per 100 g in a recipe using the following ingredients:",
        "Compute the nutritional profile per 100 g in a recipe made with these ingredients:",
        "Assess the nutrient values per 100 grams in a recipe that includes the following ingredients:",
        "Review the nutritional values per 100 grams in a recipe prepared with these ingredients:",
        "Check the nutrient profile per 100 g in a recipe using these ingredients:",
        "Verify the nutrient values per 100 g in a recipe containing the following ingredients:",
        "Gauge the nutritional breakdown per 100 grams in a recipe with the following ingredients:",
        "Determine the nutritional profile per 100 grams in a recipe that uses these ingredients:",
        "Calculate the nutritional composition per 100 g in a recipe containing these ingredients:",
        "Find the nutrient values per 100 g in a recipe with these ingredients:",
        "Identify the nutrient content per 100 grams in a recipe that includes the following ingredients:",
        "Ascertain the nutrient values per 100 grams in a recipe that has these ingredients:",
        "Discover the nutrient profile per 100 g in a recipe containing these ingredients:",
        "Establish the nutrient breakdown per 100 g in a recipe with the following ingredients:",
        "Evaluate the nutritional profile per 100 grams in a recipe using these ingredients:",
        "Analyze the nutrient values per 100 grams in a recipe made with these ingredients:",
        "Compute the nutrient breakdown per 100 g in a recipe that includes these ingredients:",
        "Assess the nutritional profile per 100 g in a recipe containing these ingredients:",
        "Review the nutrient values per 100 grams in a recipe with these ingredients:",
        "Check the nutritional content per 100 grams in a recipe that uses the following ingredients:",
        "Verify the nutrient profile per 100 g in a recipe containing these ingredients:",
        "Gauge the nutrient content per 100 g in a recipe that comprises these ingredients:",
        "Determine the nutrient profile per 100 grams in a recipe with these ingredients:",
        "Calculate the nutrient values per 100 grams in a recipe that includes these ingredients:",
        "Find the nutrient profile per 100 g in a recipe that contains these ingredients:",
        "Identify the nutrient composition per 100 g in a recipe using the following ingredients:",
        "Ascertain the nutrient values per 100 grams in a recipe that includes these ingredients:",
        "Discover the nutritional content per 100 grams in a recipe with these ingredients:",
        "Establish the nutrient values per 100 g in a recipe containing the following ingredients:",
        "Evaluate the nutritional breakdown per 100 g in a recipe that includes these ingredients:",
        "Analyze the nutrient composition per 100 grams in a recipe using these ingredients:",
        "Compute the nutrient values per 100 grams in a recipe with the following ingredients:",
        "Assess the nutritional content per 100 g in a recipe containing these ingredients:",
        "Review the nutrient values per 100 g in a recipe that comprises these ingredients:",
        "Check the nutrient values per 100 grams in a recipe that includes the following ingredients:",
        "Verify the nutritional profile per 100 grams in a recipe made with these ingredients:",
        "Gauge the nutritional values per 100 g in a recipe prepared with these ingredients:",
        "Determine the nutrient content per 100 g in a recipe that has these ingredients:",
        "Calculate the nutritional profile per 100 grams in a recipe that contains these ingredients:",
        "Find the nutrient breakdown per 100 grams in a recipe using the following ingredients:",
        "Identify the nutrient values per 100 g in a recipe that includes these ingredients:",
        "Ascertain the nutritional profile per 100 g in a recipe with the following ingredients:",
        "Discover the nutrient breakdown per 100 grams in a recipe that comprises these ingredients:",
        "Establish the nutritional content per 100 grams in a recipe that contains these ingredients:",
        "Evaluate the nutrient profile per 100 g in a recipe made with these ingredients:",
        "Analyze the nutritional values per 100 g in a recipe that has these ingredients:",
        "Compute the nutrient breakdown per 100 grams in a recipe using these ingredients:",
        "Assess the nutrient profile per 100 grams in a recipe containing the following ingredients:",
        "Review the nutritional composition per 100 g in a recipe with these ingredients:",
        "Check the nutrient profile per 100 g in a recipe that includes these ingredients:",
        "Verify the nutrient composition per 100 grams in a recipe that uses these ingredients:",
        "Gauge the nutrient breakdown per 100 grams in a recipe that includes the following ingredients:",
        "Determine the nutritional values per 100 g in a recipe that contains these ingredients:",
        "Calculate the nutrient content per 100 g in a recipe with the following ingredients:",
        "Find the nutrient profile per 100 grams in a recipe that includes these ingredients:",
        "Identify the nutritional composition per 100 grams in a recipe with these ingredients:",
        "Ascertain the nutritional values per 100 g in a recipe that contains these ingredients:",
        "Discover the nutrient values per 100 g in a recipe prepared with these ingredients:",
        "Establish the nutrient breakdown per 100 grams in a recipe using the following ingredients:",
        "Evaluate the nutritional profile per 100 grams in a recipe that includes these ingredients:",
        "Analyze the nutrient composition per 100 g in a recipe containing these ingredients:",
        "Compute the nutritional values per 100 g in a recipe with these ingredients:",
        "Assess the nutrient breakdown per 100 grams in a recipe that includes the following ingredients:",
        "Review the nutrient values per 100 grams in a recipe using these ingredients:",
        "Check the nutritional values per 100 g in a recipe that comprises these ingredients:",
        "Verify the nutrient content per 100 g in a recipe that has these ingredients:",
        "Gauge the nutrient composition per 100 grams in a recipe with the following ingredients:",
        "Determine the nutrient values per 100 grams in a recipe that uses these ingredients:",
        "Calculate the nutritional values per 100 g in a recipe containing the following ingredients:",
        "Find the nutritional profile per 100 g in a recipe with the following ingredients:",
        "Identify the nutrient breakdown per 100 grams in a recipe prepared with these ingredients:",
        "Ascertain the nutrient values per 100 grams in a recipe using these ingredients:",
        "Discover the nutritional values per 100 g in a recipe that includes these ingredients:",
        "Establish the nutrient composition per 100 g in a recipe that comprises these ingredients:",
        "Evaluate the nutrient values per 100 grams in a recipe that has these ingredients:",
        "Analyze the nutritional breakdown per 100 grams in a recipe with the following ingredients:",
        "Compute the nutrient values per 100 g in a recipe that contains these ingredients:",
        "Assess the nutrient profile per 100 g in a recipe that includes these ingredients:",
        "Review the nutrient composition per 100 grams in a recipe using the following ingredients:",
        "Check the nutrient values per 100 grams in a recipe that comprises these ingredients:",
        "Verify the nutritional profile per 100 g in a recipe with these ingredients:",
        "Gauge the nutrient profile per 100 g in a recipe using these ingredients:",
        "Determine the nutrient values per 100 grams in a recipe that has these ingredients:",
        "Calculate the nutrient profile per 100 grams in a recipe containing the following ingredients:",
        "Find the nutritional breakdown per 100 g in a recipe that includes these ingredients:",
        "Identify the nutritional profile per 100 g in a recipe that uses these ingredients:",
        "Ascertain the nutrient values per 100 grams in a recipe with the following ingredients:",
        "Discover the nutrient values per 100 grams in a recipe that includes these ingredients:",
        "Establish the nutrient values per 100 g in a recipe using these ingredients:",
        "Evaluate the nutritional profile per 100 g in a recipe that has these ingredients:",
        "Analyze the nutrient values per 100 grams in a recipe with the following ingredients:",
        "Compute the nutrient values per 100 grams in a recipe that comprises these ingredients:",
        "Assess the nutrient profile per 100 g in a recipe that includes the following ingredients:",
        "Review the nutrient profile per 100 g in a recipe containing these ingredients:",
        "Check the nutrient values per 100 grams in a recipe using these ingredients:",
        "Verify the nutritional breakdown per 100 grams in a recipe that has these ingredients:",
        "Gauge the nutrient values per 100 g in a recipe containing the following ingredients:",
        "Determine the nutritional values per 100 g in a recipe that comprises these ingredients:",
        "Calculate the nutrient values per 100 grams in a recipe with these ingredients:",
        "Find the nutrient profile per 100 grams in a recipe using these ingredients:",
        "Identify the nutritional values per 100 g in a recipe that includes the following ingredients:",
        "Ascertain the nutrient breakdown per 100 g in a recipe with these ingredients:",
        "Discover the nutrient profile per 100 grams in a recipe that contains the following ingredients:",
        "Establish the nutritional breakdown per 100 grams in a recipe that includes these ingredients:",
        "Evaluate the nutrient composition per 100 g in a recipe that has these ingredients:",
        "Analyze the nutrient profile per 100 g in a recipe with these ingredients:",
        "Compute the nutritional content per 100 grams in a recipe that includes the following ingredients:",
        "Assess the nutrient values per 100 grams in a recipe with these ingredients:",
        "Review the nutritional values per 100 g in a recipe containing these ingredients:",
        "Check the nutrient breakdown per 100 g in a recipe that includes these ingredients:",
        "Verify the nutritional content per 100 grams in a recipe that comprises these ingredients:",
        "Gauge the nutrient values per 100 grams in a recipe prepared with the following ingredients:",
        "Determine the nutrient values per 100 g in a recipe that uses the following ingredients:",
        "Calculate the nutritional values per 100 g in a recipe with these ingredients:",
        "Find the nutrient composition per 100 grams in a recipe containing these ingredients:",
        "Identify the nutrient values per 100 grams in a recipe that includes the following ingredients:",
        "Ascertain the nutrient values per 100 g in a recipe with these ingredients:",
        "Discover the nutrient breakdown per 100 g in a recipe prepared with these ingredients:",
        "Establish the nutrient profile per 100 grams in a recipe that includes the following ingredients:",
        "Evaluate the nutrient values per 100 grams in a recipe with these ingredients:",
        "Analyze the nutrient values per 100 g in a recipe that uses these ingredients:",
        "Compute the nutritional profile per 100 g in a recipe that comprises these ingredients:",
        "Assess the nutrient values per 100 grams in a recipe with the following ingredients:",
        "Review the nutrient breakdown per 100 grams in a recipe containing these ingredients:",
        "Check the nutritional values per 100 g in a recipe that includes these ingredients:",
        "Verify the nutrient breakdown per 100 g in a recipe prepared with these ingredients:",
        "Gauge the nutrient values per 100 g in a recipe that has the following ingredients:"
    ]
    recipe_answer_prompts = [
        "The nutritional values per 100 g are:",
        "Nutrient content per 100 g:",
        "Per 100 g, the nutrient values are:",
        "Per 100 grams, the nutrient values are:",
        "The nutrient content for every 100 g is:",
        "Nutritional values for each 100 g:",
        "Each 100 g contains the following nutrient values:",
        "Nutrient breakdown per 100 g:",
        "The nutrient profile per 100 g is:",
        "Nutrient values in 100 g:",
        "Nutritional information per 100 g:",
        "Values per 100 g of nutrients:",
        "Nutrient facts per 100 g:",
        "For every 100 g, the nutrients are:",
        "Each 100 grams has these nutrient values:",
        "Nutrient content in 100 g:",
        "Nutrient values in each 100 g:",
        "Nutritional breakdown per 100 g:",
        "Nutrient values per 100 grams:",
        "The nutritional profile per 100 g is:",
        "Nutrient details per 100 g:",
        "Nutritional data per 100 g:",
        "Nutrient information per 100 g:",
        "The nutrients per 100 g are:",
        "Each 100 g provides these nutrient values:",
        "The nutrient values in 100 g are:",
        "In 100 g, the nutrient values are:",
        "For 100 g, the nutrient values are:",
        "Nutritional content for every 100 g:",
        "Nutrient breakdown in 100 g:",
        "Nutrient values per 100 g listed:",
        "Every 100 g has these nutrient values:",
        "Per 100 g nutrient breakdown:",
        "Nutrient details in 100 g:",
        "Nutrient profile for each 100 g:",
        "The nutrient values found in 100 g are:",
        "In each 100 g, the nutrient values are:",
        "Nutrient facts in 100 g:",
        "Nutritional content per 100 grams:",
        "Nutritional breakdown in 100 g:",
        "Values of nutrients per 100 g:",
        "The nutrient values for 100 g are:",
        "Nutritional profile per 100 grams:",
        "Nutrient values in every 100 g:",
        "Nutrient information in 100 g:",
        "Per 100 g, the nutrients are:",
        "Nutritional details per 100 g:",
        "Nutrient values in each 100 grams:",
        "The nutrient values per 100 grams are:",
        "Every 100 grams contains the following nutrient values:",
        "Nutrient content per 100 grams:",
        "Nutrient values in per 100 g:",
        "The nutrient content for each 100 g is:",
        "For every 100 g, the nutrient values are:",
        "Each 100 g contains these nutrient values:",
        "Nutrient values in 100 grams:",
        "Nutrient profile in 100 g:",
        "Nutritional values in 100 g:",
        "Per 100 g, the nutritional values are:",
        "The nutrient breakdown per 100 g is:",
        "Nutritional facts per 100 g:",
        "Nutrient details per 100 grams:",
        "Nutrient content in each 100 g:",
        "For 100 grams, the nutrient values are:",
        "Nutritional profile in 100 grams:",
        "Nutrient values listed per 100 g:",
        "Each 100 grams provides these nutrient values:",
        "In every 100 g, the nutrient values are:",
        "Nutritional values per 100 grams:",
        "The nutrient values in each 100 g are:",
        "Nutrient values in every 100 grams:",
        "Values of nutrients in 100 g:",
        "Nutritional information in 100 g:",
        "Each 100 g has these nutrient values:",
        "Nutrient breakdown per 100 grams:",
        "Nutrient facts per 100 grams:",
        "Nutritional breakdown for every 100 g:",
        "Nutrient profile per 100 g listed:",
        "The nutrient content per 100 grams is:",
        "Nutritional values found in 100 g:",
        "For every 100 grams, the nutrient values are:",
        "Nutritional values in each 100 g:",
        "Nutrient details in 100 grams:",
        "In 100 grams, the nutrient values are:",
        "Each 100 g provides the following nutrient values:",
        "Nutrient values per 100 g given:",
        "Nutritional values listed per 100 g:",
        "The nutrients in 100 g are:",
        "Nutrient breakdown for each 100 g:",
        "Nutrient profile per 100 grams:",
        "Nutrient values per 100 g presented:",
        "The nutrient values for each 100 g are:",
        "Nutritional details in 100 g:",
        "Every 100 g has the following nutrient values:",
        "Nutrient content for each 100 g:",
        "Per 100 grams nutrient breakdown:",
        "Nutrient profile in each 100 g:",
        "Nutritional facts in 100 g:",
        "Values of nutrients per 100 grams:",
        "The nutrient values in 100 grams are:",
        "Nutrient details for every 100 g:",
        "Nutrient values for 100 g listed:",
        "Each 100 g contains the following nutrient values:",
        "Nutrient information in each 100 g:",
        "The nutrient values per 100 g listed are:",
        "Nutrient values per 100 grams given:",
        "For each 100 g, the nutrient values are:",
        "The nutrient content per 100 g listed is:",
        "Nutritional content in 100 g:",
        "Nutritional values per 100 g provided:",
        "Nutrient values per 100 g detailed:",
        "Nutrient values listed in 100 g:",
        "The nutrients for each 100 g are:",
        "Nutrient values per 100 g found:",
        "Per 100 grams, the nutritional values are:",
        "Nutritional breakdown for 100 grams:",
        "Nutrient values per 100 g shown:",
        "Nutritional values for each 100 grams:",
        "Each 100 grams has these nutrient values:",
        "Nutrient profile for every 100 g:",
        "Nutrient values per 100 grams listed:",
        "Nutritional values for 100 g are:",
        "Nutrient values for each 100 grams:",
        "Nutrient content per 100 g detailed:",
        "The nutritional content per 100 g is:",
        "Nutrient values in each 100 grams are:",
        "Per 100 grams the nutrient values are:",
        "Nutrient values per 100 g provided:",
        "Nutritional values given per 100 g:",
        "Nutrient values for 100 grams:",
        "Nutrient profile detailed per 100 g:",
        "The nutrient values in every 100 g are:",
        "Nutrient values per 100 g highlighted:",
        "Nutrient values detailed for 100 g:",
        "Nutritional values in every 100 g:",
        "Nutrient values detailed per 100 grams:",
        "Each 100 g has these nutritional values:",
        "Nutrient values shown for 100 g:",
        "Nutrient values highlighted per 100 g:",
        "The nutrient values detailed for 100 g are:",
        "Nutrient values highlighted for 100 grams:",
        "The nutrient values provided per 100 g are:",
        "Nutrient values provided per 100 grams:",
        "Nutrient values per 100 g listed above:",
        "The nutrient values per 100 grams given are:",
        "Nutrient values per 100 grams shown:",
        "Each 100 grams provides the following nutrient values:",
        "The nutrient values highlighted per 100 g are:"
    ]

    def generate(self, sample: list, ingredients: list) -> dict:
        questions = []
        answers = []

        for i, recipe, _, _ in sample:
            question = ''

            for ingredient, quantity, unit in zip(recipe.ingredients, recipe.ingredient_quantity,
                                                  recipe.ingredient_unit):
                question += f'{quantity} {unit} {ingredient}, '
            question = question[:-2]

            answer = (f'energy - {recipe.energy_value:.2f}, fat - {recipe.fat_value:.2f}, '
                      f'protein - {recipe.protein_value:.2f}, salt - {recipe.salt_value:.2f}, '
                      f'saturates - {recipe.saturates_value:.2f}, sugars - {recipe.sugars_value:.2f}')

            question_prompt = self.recipe_question_prompts[
                self.generate_random_number(0, len(self.recipe_question_prompts) - 1)]

            questions.append(
                f'{self.start} '
                f'{question_prompt.replace('a recipe', f'{recipe.title}')}'
                f' {question.strip()} {self.end}')
            answers.append(
                f'{self.recipe_answer_prompts[
                    self.generate_random_number(0, len(self.recipe_answer_prompts) - 1)]}'
                f' {answer}')

        return {'questions': questions, 'answers': answers}


class RecipeIngrFsaSequenceGenerator(RecipeSequenceGenerator):
    fsa_lights_question_prompts = [
        "Determine the fsa lights per 100 g in a recipe that includes these ingredients:",
        "Calculate the fsa traffic lights per 100 g in a recipe with the following ingredients:",
        "Find the Food Standards Agency traffic lights per 100 grams in a recipe containing these ingredients:",
        "Identify the fsa traffic lights per 100 grams for a recipe with the following ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 g in a recipe using these ingredients:",
        "Discover the FSA lights per 100 g in a recipe that uses the following ingredients:",
        "Establish the Food Standards Agency traffic lights per 100 grams in a recipe that contains these ingredients:",
        "Evaluate the FSA lights per 100 grams in a recipe made with these ingredients:",
        "Analyze the Food Standards Agency traffic lights per 100 g in a recipe with the following ingredients:",
        "Compute the FSA lights per 100 g in a recipe that includes these ingredients:",
        "Assess the Food Standards Agency traffic lights per 100 grams in a recipe containing the following ingredients:",
        "Review the FSA lights per 100 grams in a recipe that comprises these ingredients:",
        "Check the Food Standards Agency traffic lights per 100 g in a recipe that has these ingredients:",
        "Verify the FSA lights per 100 g in a recipe prepared with these ingredients:",
        "Gauge the Food Standards Agency traffic lights per 100 grams in a recipe that incorporates the following ingredients:",
        "Determine the fsa traffic lights per 100 grams in a recipe containing these ingredients:",
        "Calculate the fsa traffic lights per 100 g in a recipe with these ingredients:",
        "Find the FSA traffic light labels per 100 g in a recipe that uses the following ingredients:",
        "Identify the FSA traffic light labels per 100 grams in a recipe containing these ingredients:",
        "Ascertain the fsa lights per 100 grams in a recipe with the following ingredients:",
        "Discover the fsa traffic lights per 100 g in a recipe that includes these ingredients:",
        "Establish the fsa traffic lights per 100 g in a recipe with these ingredients:",
        "Evaluate the fsa traffic lights per 100 grams in a recipe using the following ingredients:",
        "Analyze the FSA traffic light labels per 100 grams in a recipe made with these ingredients:",
        "Compute the fsa lights per 100 g in a recipe containing these ingredients:",
        "Assess the FSA traffic light labels per 100 g in a recipe with the following ingredients:",
        "Review the FSA traffic light labels per 100 grams in a recipe that comprises these ingredients:",
        "Check the fsa traffic lights per 100 grams in a recipe that includes these ingredients:",
        "Verify the fsa traffic lights per 100 g in a recipe that contains the following ingredients:",
        "Gauge the fsa traffic lights per 100 g in a recipe made with these ingredients:",
        "Determine the Food Standards Agency traffic lights per 100 grams in a recipe using the following ingredients:",
        "Calculate the fsa lights per 100 grams in a recipe with these ingredients:",
        "Find the fsa traffic lights per 100 g in a recipe containing the following ingredients:",
        "Identify the fsa lights per 100 g in a recipe prepared with these ingredients:",
        "Ascertain the FSA traffic light labels per 100 grams in a recipe that includes these ingredients:",
        "Discover the FSA traffic light labels per 100 grams in a recipe with these ingredients:",
        "Establish the fsa lights per 100 g in a recipe containing these ingredients:",
        "Evaluate the fsa traffic lights per 100 g in a recipe that uses the following ingredients:",
        "Analyze the FSA lights per 100 grams in a recipe made with these ingredients:",
        "Compute the fsa lights per 100 grams in a recipe that includes these ingredients:",
        "Assess the fsa traffic lights per 100 g in a recipe using the following ingredients:",
        "Review the fsa traffic lights per 100 g in a recipe containing these ingredients:",
        "Check the FSA traffic light labels per 100 g in a recipe with the following ingredients:",
        "Verify the FSA traffic light labels per 100 grams in a recipe that comprises these ingredients:",
        "Gauge the Food Standards Agency traffic lights per 100 grams in a recipe that has these ingredients:",
        "Determine the FSA traffic light labels per 100 g in a recipe made with these ingredients:",
        "Calculate the fsa traffic lights per 100 g in a recipe using the following ingredients:",
        "Find the fsa traffic lights per 100 grams in a recipe with the following ingredients:",
        "Identify the fsa traffic lights per 100 grams in a recipe that includes these ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 g in a recipe that comprises these ingredients:",
        "Discover the fsa traffic lights per 100 g in a recipe that has these ingredients:",
        "Establish the fsa lights per 100 grams in a recipe with the following ingredients:",
        "Evaluate the fsa traffic lights per 100 grams in a recipe that contains these ingredients:",
        "Analyze the fsa lights per 100 g in a recipe using the following ingredients:",
        "Compute the FSA traffic light labels per 100 g in a recipe made with these ingredients:",
        "Assess the Food Standards Agency traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Review the fsa lights per 100 grams in a recipe prepared with these ingredients:",
        "Check the fsa lights per 100 g in a recipe using these ingredients:",
        "Verify the Food Standards Agency traffic lights per 100 g in a recipe containing the following ingredients:",
        "Gauge the FSA traffic light labels per 100 grams in a recipe with the following ingredients:",
        "Determine the FSA traffic light labels per 100 grams in a recipe that uses these ingredients:",
        "Calculate the FSA lights per 100 g in a recipe containing these ingredients:",
        "Find the Food Standards Agency traffic lights per 100 g in a recipe with these ingredients:",
        "Identify the fsa traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 grams in a recipe that has these ingredients:",
        "Discover the fsa lights per 100 g in a recipe containing these ingredients:",
        "Establish the fsa traffic lights per 100 g in a recipe with the following ingredients:",
        "Evaluate the FSA traffic light labels per 100 grams in a recipe using these ingredients:",
        "Analyze the Food Standards Agency traffic lights per 100 grams in a recipe made with these ingredients:",
        "Compute the fsa traffic lights per 100 g in a recipe that includes these ingredients:",
        "Assess the FSA traffic light labels per 100 g in a recipe containing these ingredients:",
        "Review the Food Standards Agency traffic lights per 100 grams in a recipe with these ingredients:",
        "Check the fsa traffic lights per 100 grams in a recipe that uses the following ingredients:",
        "Verify the fsa lights per 100 g in a recipe containing these ingredients:",
        "Gauge the fsa traffic lights per 100 g in a recipe that comprises these ingredients:",
        "Determine the fsa lights per 100 grams in a recipe with these ingredients:",
        "Calculate the Food Standards Agency traffic lights per 100 grams in a recipe that includes these ingredients:",
        "Find the fsa lights per 100 g in a recipe that contains these ingredients:",
        "Identify the fsa traffic lights per 100 g in a recipe using the following ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 grams in a recipe that includes these ingredients:",
        "Discover the fsa traffic lights per 100 grams in a recipe with these ingredients:",
        "Establish the Food Standards Agency traffic lights per 100 g in a recipe containing the following ingredients:",
        "Evaluate the FSA traffic light labels per 100 g in a recipe that includes these ingredients:",
        "Analyze the fsa traffic lights per 100 grams in a recipe using these ingredients:",
        "Compute the Food Standards Agency traffic lights per 100 grams in a recipe with the following ingredients:",
        "Assess the fsa traffic lights per 100 g in a recipe containing these ingredients:",
        "Review the FSA lights per 100 g in a recipe that comprises these ingredients:",
        "Check the Food Standards Agency traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Verify the FSA traffic light labels per 100 grams in a recipe made with these ingredients:",
        "Gauge the fsa lights per 100 g in a recipe prepared with these ingredients:",
        "Determine the fsa traffic lights per 100 g in a recipe that has these ingredients:",
        "Calculate the FSA traffic light labels per 100 grams in a recipe that contains these ingredients:",
        "Find the fsa traffic lights per 100 grams in a recipe using the following ingredients:",
        "Identify the Food Standards Agency traffic lights per 100 g in a recipe that includes these ingredients:",
        "Ascertain the FSA traffic light labels per 100 g in a recipe with the following ingredients:",
        "Discover the fsa traffic lights per 100 grams in a recipe that comprises these ingredients:",
        "Establish the fsa traffic lights per 100 grams in a recipe that contains these ingredients:",
        "Evaluate the fsa lights per 100 g in a recipe made with these ingredients:",
        "Analyze the fsa lights per 100 g in a recipe that has these ingredients:",
        "Compute the fsa traffic lights per 100 grams in a recipe using these ingredients:",
        "Assess the fsa lights per 100 grams in a recipe containing the following ingredients:",
        "Review the FSA lights per 100 g in a recipe with these ingredients:",
        "Check the fsa lights per 100 g in a recipe that includes these ingredients:",
        "Verify the fsa traffic lights per 100 grams in a recipe that uses these ingredients:",
        "Gauge the fsa traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Determine the fsa lights per 100 g in a recipe that contains these ingredients:",
        "Calculate the fsa traffic lights per 100 g in a recipe with the following ingredients:",
        "Find the fsa lights per 100 grams in a recipe that includes these ingredients:",
        "Identify the FSA lights per 100 grams in a recipe with these ingredients:",
        "Ascertain the fsa lights per 100 g in a recipe that contains these ingredients:",
        "Discover the Food Standards Agency traffic lights per 100 g in a recipe prepared with these ingredients:",
        "Establish the fsa traffic lights per 100 grams in a recipe using the following ingredients:",
        "Evaluate the FSA traffic light labels per 100 grams in a recipe that includes these ingredients:",
        "Analyze the fsa traffic lights per 100 g in a recipe containing these ingredients:",
        "Compute the fsa lights per 100 g in a recipe with these ingredients:",
        "Assess the fsa traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Review the Food Standards Agency traffic lights per 100 grams in a recipe using these ingredients:",
        "Check the fsa lights per 100 g in a recipe that comprises these ingredients:",
        "Verify the fsa traffic lights per 100 g in a recipe that has these ingredients:",
        "Gauge the fsa traffic lights per 100 grams in a recipe with the following ingredients:",
        "Determine the Food Standards Agency traffic lights per 100 grams in a recipe that uses these ingredients:",
        "Calculate the fsa lights per 100 g in a recipe containing the following ingredients:",
        "Find the FSA traffic light labels per 100 g in a recipe with the following ingredients:",
        "Identify the fsa traffic lights per 100 grams in a recipe prepared with these ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 grams in a recipe using these ingredients:",
        "Discover the fsa lights per 100 g in a recipe that includes these ingredients:",
        "Establish the fsa traffic lights per 100 g in a recipe that comprises these ingredients:",
        "Evaluate the Food Standards Agency traffic lights per 100 grams in a recipe that has these ingredients:",
        "Analyze the FSA traffic light labels per 100 grams in a recipe with the following ingredients:",
        "Compute the Food Standards Agency traffic lights per 100 g in a recipe that contains these ingredients:",
        "Assess the fsa lights per 100 g in a recipe that includes these ingredients:",
        "Review the fsa traffic lights per 100 grams in a recipe using the following ingredients:",
        "Check the Food Standards Agency traffic lights per 100 grams in a recipe that comprises these ingredients:",
        "Verify the FSA traffic light labels per 100 g in a recipe with these ingredients:",
        "Gauge the fsa lights per 100 g in a recipe using these ingredients:",
        "Determine the Food Standards Agency traffic lights per 100 grams in a recipe that has these ingredients:",
        "Calculate the fsa lights per 100 grams in a recipe containing the following ingredients:",
        "Find the FSA traffic light labels per 100 g in a recipe that includes these ingredients:",
        "Identify the FSA traffic light labels per 100 g in a recipe that uses these ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 grams in a recipe with the following ingredients:",
        "Discover the Food Standards Agency traffic lights per 100 grams in a recipe that includes these ingredients:",
        "Establish the Food Standards Agency traffic lights per 100 g in a recipe using these ingredients:",
        "Evaluate the FSA traffic light labels per 100 g in a recipe that has these ingredients:",
        "Analyze the Food Standards Agency traffic lights per 100 grams in a recipe with the following ingredients:",
        "Compute the Food Standards Agency traffic lights per 100 grams in a recipe that comprises these ingredients:",
        "Assess the fsa lights per 100 g in a recipe that includes the following ingredients:",
        "Review the fsa lights per 100 g in a recipe containing these ingredients:",
        "Check the Food Standards Agency traffic lights per 100 grams in a recipe using these ingredients:",
        "Verify the FSA traffic light labels per 100 grams in a recipe that has these ingredients:",
        "Gauge the Food Standards Agency traffic lights per 100 g in a recipe containing the following ingredients:",
        "Determine the fsa lights per 100 g in a recipe that comprises these ingredients:",
        "Calculate the Food Standards Agency traffic lights per 100 grams in a recipe with these ingredients:",
        "Find the fsa lights per 100 grams in a recipe using these ingredients:",
        "Identify the fsa lights per 100 g in a recipe that includes the following ingredients:",
        "Ascertain the fsa traffic lights per 100 g in a recipe with these ingredients:",
        "Discover the fsa lights per 100 grams in a recipe that contains the following ingredients:",
        "Establish the FSA traffic light labels per 100 grams in a recipe that includes these ingredients:",
        "Evaluate the fsa traffic lights per 100 g in a recipe that has these ingredients:",
        "Analyze the fsa lights per 100 g in a recipe with these ingredients:",
        "Compute the fsa traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Assess the Food Standards Agency traffic lights per 100 grams in a recipe with these ingredients:",
        "Review the fsa lights per 100 g in a recipe containing these ingredients:",
        "Check the fsa traffic lights per 100 g in a recipe that includes these ingredients:",
        "Verify the fsa traffic lights per 100 grams in a recipe that comprises these ingredients:",
        "Gauge the Food Standards Agency traffic lights per 100 grams in a recipe prepared with the following ingredients:",
        "Determine the Food Standards Agency traffic lights per 100 g in a recipe that uses the following ingredients:",
        "Calculate the fsa lights per 100 g in a recipe with these ingredients:",
        "Find the fsa traffic lights per 100 grams in a recipe containing these ingredients:",
        "Identify the Food Standards Agency traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 g in a recipe with these ingredients:",
        "Discover the fsa traffic lights per 100 g in a recipe prepared with these ingredients:",
        "Establish the fsa lights per 100 grams in a recipe that includes the following ingredients:",
        "Evaluate the Food Standards Agency traffic lights per 100 grams in a recipe with these ingredients:",
        "Analyze the Food Standards Agency traffic lights per 100 g in a recipe that uses these ingredients:",
        "Compute the FSA traffic light labels per 100 g in a recipe that comprises these ingredients:",
        "Assess the Food Standards Agency traffic lights per 100 grams in a recipe with the following ingredients:",
        "Review the fsa traffic lights per 100 grams in a recipe containing these ingredients:",
        "Check the fsa lights per 100 g in a recipe that includes these ingredients:",
        "Verify the fsa traffic lights per 100 g in a recipe prepared with these ingredients:",
        "Gauge the Food Standards Agency traffic lights per 100 g in a recipe that has the following ingredients:"
    ]
    fsa_lights_answer_prompts = [
        'The fsa lights per 100 g are:',
        'Fsa traffic lights per 100 g:',
        'Per 100 g, the Food Standards Agency traffic lights are:',
        'Per 100 grams, the Food Standards Agency traffic lights are:',
        'The fsa traffic lights for every 100 g are:',
        'Fsa lights for each 100 g:',
        'Each 100 g contains the following Food Standards Agency traffic lights:',
        'Food Standards Agency lights per 100 g:',
        'The Food Standards Agency lights per 100 g are:',
        'Food Standards Agency traffic lights in 100 g:',
        'FSA traffic light labels per 100 g:',
        'Values per 100 g of FSA traffic lights:',
        'Food Standards Agency traffic lights per 100 g:',
        'For every 100 g, the FSA traffic lights are:',
        'Each 100 grams has these Food Standards Agency traffic lights:',
        'Fsa traffic lights in 100 g:',
        'Food Standards Agency traffic lights in each 100 g:',
        'FSA traffic light labels per 100 g:',
        'Food Standards Agency traffic lights per 100 grams:',
        'The FSA traffic light labels per 100 g are:',
        'FSA lights per 100 g:',
        'FSA lights per 100 g:',
        'FSA traffic light labels per 100 g:',
        'The FSA traffic lights per 100 g are:',
        'Each 100 g provides these Food Standards Agency traffic lights:',
        'The Food Standards Agency traffic lights in 100 g are:',
        'In 100 g, the Food Standards Agency traffic lights are:',
        'For 100 g, the Food Standards Agency traffic lights are:',
        'FSA lights for every 100 g:',
        'Food Standards Agency lights in 100 g:',
        'Food Standards Agency traffic lights per 100 g listed:',
        'Every 100 g has these Food Standards Agency traffic lights:',
        'Per 100 g Food Standards Agency lights:',
        'FSA lights in 100 g:',
        'Food Standards Agency lights for each 100 g:',
        'The Food Standards Agency traffic lights found in 100 g are:',
        'In each 100 g, the Food Standards Agency traffic lights are:',
        'Food Standards Agency traffic lights in 100 g:',
        'FSA lights per 100 grams:',
        'FSA traffic light labels in 100 g:',
        'Values of FSA traffic lights per 100 g:',
        'The Food Standards Agency traffic lights for 100 g are:',
        'FSA traffic light labels per 100 grams:',
        'Food Standards Agency traffic lights in every 100 g:',
        'FSA traffic light labels in 100 g:',
        'Per 100 g, the FSA traffic lights are:',
        'FSA traffic light labels per 100 g:',
        'Food Standards Agency traffic lights in each 100 grams:',
        'The Food Standards Agency traffic lights per 100 grams are:',
        'Every 100 grams contains the following Food Standards Agency traffic lights:',
        'Fsa traffic lights per 100 grams:',
        'Food Standards Agency traffic lights in per 100 g:',
        'The fsa traffic lights for each 100 g are:',
        'For every 100 g, the Food Standards Agency traffic lights are:',
        'Each 100 g contains these Food Standards Agency traffic lights:',
        'Food Standards Agency traffic lights in 100 grams:',
        'Food Standards Agency lights in 100 g:',
        'FSA lights in 100 g:',
        'Per 100 g, the fsa lights are:',
        'The Food Standards Agency lights per 100 g are:',
        'FSA lights per 100 g:',
        'FSA lights per 100 grams:',
        'Fsa traffic lights in each 100 g:',
        'For 100 grams, the Food Standards Agency traffic lights are:',
        'FSA traffic light labels in 100 grams:',
        'Food Standards Agency traffic lights listed per 100 g:',
        'Each 100 grams provides these Food Standards Agency traffic lights:',
        'In every 100 g, the Food Standards Agency traffic lights are:',
        'FSA lights per 100 grams:',
        'The Food Standards Agency traffic lights in each 100 g are:',
        'Food Standards Agency traffic lights in every 100 grams:',
        'Values of FSA traffic lights in 100 g:',
        'FSA traffic light labels in 100 g:',
        'Each 100 g has these Food Standards Agency traffic lights:',
        'Food Standards Agency lights per 100 grams:',
        'Food Standards Agency traffic lights per 100 grams:',
        'FSA traffic light labels for every 100 g:',
        'Food Standards Agency lights per 100 g listed:',
        'The fsa traffic lights per 100 grams are:',
        'Fsa lights found in 100 g:',
        'For every 100 grams, the Food Standards Agency traffic lights are:',
        'Fsa lights in each 100 g:',
        'FSA lights in 100 grams:',
        'In 100 grams, the Food Standards Agency traffic lights are:',
        'Each 100 g provides the following Food Standards Agency traffic lights:',
        'Food Standards Agency traffic lights per 100 g given:',
        'Fsa lights listed per 100 g:',
        'The FSA traffic lights in 100 g are:',
        'Food Standards Agency lights for each 100 g:',
        'Food Standards Agency lights per 100 grams:',
        'Food Standards Agency traffic lights per 100 g presented:',
        'The Food Standards Agency traffic lights for each 100 g are:',
        'FSA traffic light labels in 100 g:',
        'Every 100 g has the following Food Standards Agency traffic lights:',
        'Fsa traffic lights for each 100 g:',
        'Per 100 grams Food Standards Agency lights:',
        'Food Standards Agency lights in each 100 g:',
        'FSA lights in 100 g:',
        'Values of FSA traffic lights per 100 grams:',
        'The Food Standards Agency traffic lights in 100 grams are:',
        'FSA lights for every 100 g:',
        'Food Standards Agency traffic lights for 100 g listed:',
        'Each 100 g contains the following Food Standards Agency traffic lights:',
        'FSA traffic light labels in each 100 g:',
        'The Food Standards Agency traffic lights per 100 g listed are:',
        'Food Standards Agency traffic lights per 100 grams given:',
        'For each 100 g, the Food Standards Agency traffic lights are:',
        'The fsa traffic lights per 100 g listed are:',
        'FSA lights in 100 g:',
        'Fsa lights per 100 g provided:',
        'Food Standards Agency traffic lights per 100 g detailed:',
        'Food Standards Agency traffic lights listed in 100 g:',
        'The FSA traffic lights for each 100 g are:',
        'Food Standards Agency traffic lights per 100 g found:',
        'Per 100 grams, the fsa lights are:',
        'FSA traffic light labels for 100 grams:',
        'Food Standards Agency traffic lights per 100 g shown:',
        'Fsa lights for each 100 grams:',
        'Each 100 grams has these Food Standards Agency traffic lights:',
        'Food Standards Agency lights for every 100 g:',
        'Food Standards Agency traffic lights per 100 grams listed:',
        'Fsa lights for 100 g are:',
        'Food Standards Agency traffic lights for each 100 grams:',
        'Fsa traffic lights per 100 g detailed:',
        'The FSA lights per 100 g are:',
        'Food Standards Agency traffic lights in each 100 grams are:',
        'Per 100 grams the Food Standards Agency traffic lights are:',
        'Food Standards Agency traffic lights per 100 g provided:',
        'Fsa lights given per 100 g:',
        'Food Standards Agency traffic lights for 100 grams:',
        'Food Standards Agency lights detailed per 100 g:',
        'The Food Standards Agency traffic lights in every 100 g are:',
        'Food Standards Agency traffic lights per 100 g highlighted:',
        'Food Standards Agency traffic lights detailed for 100 g:',
        'Fsa lights in every 100 g:',
        'Food Standards Agency traffic lights detailed per 100 grams:',
        'Each 100 g has these fsa lights:',
        'Food Standards Agency traffic lights shown for 100 g:',
        'Food Standards Agency traffic lights highlighted per 100 g:',
        'The Food Standards Agency traffic lights detailed for 100 g are:',
        'Food Standards Agency traffic lights highlighted for 100 grams:',
        'The Food Standards Agency traffic lights provided per 100 g are:',
        'Food Standards Agency traffic lights provided per 100 grams:',
        'Food Standards Agency traffic lights per 100 g listed above:',
        'The Food Standards Agency traffic lights per 100 grams given are:',
        'Food Standards Agency traffic lights per 100 grams shown:',
        'Each 100 grams provides the following Food Standards Agency traffic lights:',
        'The Food Standards Agency traffic lights highlighted per 100 g are:'
    ]

    def generate(self, sample: list, ingredients: list) -> dict:
        questions = []
        answers = []

        for i, recipe, _, _ in sample:
            question = ''

            for ingredient, quantity, unit in zip(recipe.ingredients, recipe.ingredient_quantity,
                                                  recipe.ingredient_unit):
                question += f'{quantity} {unit} {ingredient}, '
            question = question[:-2]

            answer = (f'fat - {recipe.fat_fsa}, '
                      f'salt - {recipe.salt_fsa}, saturates - {recipe.saturates_fsa}, sugars - {recipe.sugars_fsa}')

            questions.append(f'{self.start} '
                             f'{self.fsa_lights_question_prompts[
                                 self.generate_random_number(0, len(self.fsa_lights_question_prompts) - 1)]}'
                             f' {question.strip()} {self.end}')
            answers.append(
                f'{self.fsa_lights_answer_prompts[self.generate_random_number(0, len(self.fsa_lights_answer_prompts) - 1)]}'
                f' {answer}')

        return {'questions': questions, 'answers': answers}


class RecipeTitlIngrFsaSequenceGenerator(RecipeSequenceGenerator):
    fsa_lights_question_prompts = [
        "Determine the fsa lights per 100 g in a recipe that includes these ingredients:",
        "Calculate the fsa traffic lights per 100 g in a recipe with the following ingredients:",
        "Find the Food Standards Agency traffic lights per 100 grams in a recipe containing these ingredients:",
        "Identify the fsa traffic lights per 100 grams for a recipe with the following ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 g in a recipe using these ingredients:",
        "Discover the FSA lights per 100 g in a recipe that uses the following ingredients:",
        "Establish the Food Standards Agency traffic lights per 100 grams in a recipe that contains these ingredients:",
        "Evaluate the FSA lights per 100 grams in a recipe made with these ingredients:",
        "Analyze the Food Standards Agency traffic lights per 100 g in a recipe with the following ingredients:",
        "Compute the FSA lights per 100 g in a recipe that includes these ingredients:",
        "Assess the Food Standards Agency traffic lights per 100 grams in a recipe containing the following ingredients:",
        "Review the FSA lights per 100 grams in a recipe that comprises these ingredients:",
        "Check the Food Standards Agency traffic lights per 100 g in a recipe that has these ingredients:",
        "Verify the FSA lights per 100 g in a recipe prepared with these ingredients:",
        "Gauge the Food Standards Agency traffic lights per 100 grams in a recipe that incorporates the following ingredients:",
        "Determine the fsa traffic lights per 100 grams in a recipe containing these ingredients:",
        "Calculate the fsa traffic lights per 100 g in a recipe with these ingredients:",
        "Find the FSA traffic light labels per 100 g in a recipe that uses the following ingredients:",
        "Identify the FSA traffic light labels per 100 grams in a recipe containing these ingredients:",
        "Ascertain the fsa lights per 100 grams in a recipe with the following ingredients:",
        "Discover the fsa traffic lights per 100 g in a recipe that includes these ingredients:",
        "Establish the fsa traffic lights per 100 g in a recipe with these ingredients:",
        "Evaluate the fsa traffic lights per 100 grams in a recipe using the following ingredients:",
        "Analyze the FSA traffic light labels per 100 grams in a recipe made with these ingredients:",
        "Compute the fsa lights per 100 g in a recipe containing these ingredients:",
        "Assess the FSA traffic light labels per 100 g in a recipe with the following ingredients:",
        "Review the FSA traffic light labels per 100 grams in a recipe that comprises these ingredients:",
        "Check the fsa traffic lights per 100 grams in a recipe that includes these ingredients:",
        "Verify the fsa traffic lights per 100 g in a recipe that contains the following ingredients:",
        "Gauge the fsa traffic lights per 100 g in a recipe made with these ingredients:",
        "Determine the Food Standards Agency traffic lights per 100 grams in a recipe using the following ingredients:",
        "Calculate the fsa lights per 100 grams in a recipe with these ingredients:",
        "Find the fsa traffic lights per 100 g in a recipe containing the following ingredients:",
        "Identify the fsa lights per 100 g in a recipe prepared with these ingredients:",
        "Ascertain the FSA traffic light labels per 100 grams in a recipe that includes these ingredients:",
        "Discover the FSA traffic light labels per 100 grams in a recipe with these ingredients:",
        "Establish the fsa lights per 100 g in a recipe containing these ingredients:",
        "Evaluate the fsa traffic lights per 100 g in a recipe that uses the following ingredients:",
        "Analyze the FSA lights per 100 grams in a recipe made with these ingredients:",
        "Compute the fsa lights per 100 grams in a recipe that includes these ingredients:",
        "Assess the fsa traffic lights per 100 g in a recipe using the following ingredients:",
        "Review the fsa traffic lights per 100 g in a recipe containing these ingredients:",
        "Check the FSA traffic light labels per 100 g in a recipe with the following ingredients:",
        "Verify the FSA traffic light labels per 100 grams in a recipe that comprises these ingredients:",
        "Gauge the Food Standards Agency traffic lights per 100 grams in a recipe that has these ingredients:",
        "Determine the FSA traffic light labels per 100 g in a recipe made with these ingredients:",
        "Calculate the fsa traffic lights per 100 g in a recipe using the following ingredients:",
        "Find the fsa traffic lights per 100 grams in a recipe with the following ingredients:",
        "Identify the fsa traffic lights per 100 grams in a recipe that includes these ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 g in a recipe that comprises these ingredients:",
        "Discover the fsa traffic lights per 100 g in a recipe that has these ingredients:",
        "Establish the fsa lights per 100 grams in a recipe with the following ingredients:",
        "Evaluate the fsa traffic lights per 100 grams in a recipe that contains these ingredients:",
        "Analyze the fsa lights per 100 g in a recipe using the following ingredients:",
        "Compute the FSA traffic light labels per 100 g in a recipe made with these ingredients:",
        "Assess the Food Standards Agency traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Review the fsa lights per 100 grams in a recipe prepared with these ingredients:",
        "Check the fsa lights per 100 g in a recipe using these ingredients:",
        "Verify the Food Standards Agency traffic lights per 100 g in a recipe containing the following ingredients:",
        "Gauge the FSA traffic light labels per 100 grams in a recipe with the following ingredients:",
        "Determine the FSA traffic light labels per 100 grams in a recipe that uses these ingredients:",
        "Calculate the FSA lights per 100 g in a recipe containing these ingredients:",
        "Find the Food Standards Agency traffic lights per 100 g in a recipe with these ingredients:",
        "Identify the fsa traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 grams in a recipe that has these ingredients:",
        "Discover the fsa lights per 100 g in a recipe containing these ingredients:",
        "Establish the fsa traffic lights per 100 g in a recipe with the following ingredients:",
        "Evaluate the FSA traffic light labels per 100 grams in a recipe using these ingredients:",
        "Analyze the Food Standards Agency traffic lights per 100 grams in a recipe made with these ingredients:",
        "Compute the fsa traffic lights per 100 g in a recipe that includes these ingredients:",
        "Assess the FSA traffic light labels per 100 g in a recipe containing these ingredients:",
        "Review the Food Standards Agency traffic lights per 100 grams in a recipe with these ingredients:",
        "Check the fsa traffic lights per 100 grams in a recipe that uses the following ingredients:",
        "Verify the fsa lights per 100 g in a recipe containing these ingredients:",
        "Gauge the fsa traffic lights per 100 g in a recipe that comprises these ingredients:",
        "Determine the fsa lights per 100 grams in a recipe with these ingredients:",
        "Calculate the Food Standards Agency traffic lights per 100 grams in a recipe that includes these ingredients:",
        "Find the fsa lights per 100 g in a recipe that contains these ingredients:",
        "Identify the fsa traffic lights per 100 g in a recipe using the following ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 grams in a recipe that includes these ingredients:",
        "Discover the fsa traffic lights per 100 grams in a recipe with these ingredients:",
        "Establish the Food Standards Agency traffic lights per 100 g in a recipe containing the following ingredients:",
        "Evaluate the FSA traffic light labels per 100 g in a recipe that includes these ingredients:",
        "Analyze the fsa traffic lights per 100 grams in a recipe using these ingredients:",
        "Compute the Food Standards Agency traffic lights per 100 grams in a recipe with the following ingredients:",
        "Assess the fsa traffic lights per 100 g in a recipe containing these ingredients:",
        "Review the FSA lights per 100 g in a recipe that comprises these ingredients:",
        "Check the Food Standards Agency traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Verify the FSA traffic light labels per 100 grams in a recipe made with these ingredients:",
        "Gauge the fsa lights per 100 g in a recipe prepared with these ingredients:",
        "Determine the fsa traffic lights per 100 g in a recipe that has these ingredients:",
        "Calculate the FSA traffic light labels per 100 grams in a recipe that contains these ingredients:",
        "Find the fsa traffic lights per 100 grams in a recipe using the following ingredients:",
        "Identify the Food Standards Agency traffic lights per 100 g in a recipe that includes these ingredients:",
        "Ascertain the FSA traffic light labels per 100 g in a recipe with the following ingredients:",
        "Discover the fsa traffic lights per 100 grams in a recipe that comprises these ingredients:",
        "Establish the fsa traffic lights per 100 grams in a recipe that contains these ingredients:",
        "Evaluate the fsa lights per 100 g in a recipe made with these ingredients:",
        "Analyze the fsa lights per 100 g in a recipe that has these ingredients:",
        "Compute the fsa traffic lights per 100 grams in a recipe using these ingredients:",
        "Assess the fsa lights per 100 grams in a recipe containing the following ingredients:",
        "Review the FSA lights per 100 g in a recipe with these ingredients:",
        "Check the fsa lights per 100 g in a recipe that includes these ingredients:",
        "Verify the fsa traffic lights per 100 grams in a recipe that uses these ingredients:",
        "Gauge the fsa traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Determine the fsa lights per 100 g in a recipe that contains these ingredients:",
        "Calculate the fsa traffic lights per 100 g in a recipe with the following ingredients:",
        "Find the fsa lights per 100 grams in a recipe that includes these ingredients:",
        "Identify the FSA lights per 100 grams in a recipe with these ingredients:",
        "Ascertain the fsa lights per 100 g in a recipe that contains these ingredients:",
        "Discover the Food Standards Agency traffic lights per 100 g in a recipe prepared with these ingredients:",
        "Establish the fsa traffic lights per 100 grams in a recipe using the following ingredients:",
        "Evaluate the FSA traffic light labels per 100 grams in a recipe that includes these ingredients:",
        "Analyze the fsa traffic lights per 100 g in a recipe containing these ingredients:",
        "Compute the fsa lights per 100 g in a recipe with these ingredients:",
        "Assess the fsa traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Review the Food Standards Agency traffic lights per 100 grams in a recipe using these ingredients:",
        "Check the fsa lights per 100 g in a recipe that comprises these ingredients:",
        "Verify the fsa traffic lights per 100 g in a recipe that has these ingredients:",
        "Gauge the fsa traffic lights per 100 grams in a recipe with the following ingredients:",
        "Determine the Food Standards Agency traffic lights per 100 grams in a recipe that uses these ingredients:",
        "Calculate the fsa lights per 100 g in a recipe containing the following ingredients:",
        "Find the FSA traffic light labels per 100 g in a recipe with the following ingredients:",
        "Identify the fsa traffic lights per 100 grams in a recipe prepared with these ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 grams in a recipe using these ingredients:",
        "Discover the fsa lights per 100 g in a recipe that includes these ingredients:",
        "Establish the fsa traffic lights per 100 g in a recipe that comprises these ingredients:",
        "Evaluate the Food Standards Agency traffic lights per 100 grams in a recipe that has these ingredients:",
        "Analyze the FSA traffic light labels per 100 grams in a recipe with the following ingredients:",
        "Compute the Food Standards Agency traffic lights per 100 g in a recipe that contains these ingredients:",
        "Assess the fsa lights per 100 g in a recipe that includes these ingredients:",
        "Review the fsa traffic lights per 100 grams in a recipe using the following ingredients:",
        "Check the Food Standards Agency traffic lights per 100 grams in a recipe that comprises these ingredients:",
        "Verify the FSA traffic light labels per 100 g in a recipe with these ingredients:",
        "Gauge the fsa lights per 100 g in a recipe using these ingredients:",
        "Determine the Food Standards Agency traffic lights per 100 grams in a recipe that has these ingredients:",
        "Calculate the fsa lights per 100 grams in a recipe containing the following ingredients:",
        "Find the FSA traffic light labels per 100 g in a recipe that includes these ingredients:",
        "Identify the FSA traffic light labels per 100 g in a recipe that uses these ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 grams in a recipe with the following ingredients:",
        "Discover the Food Standards Agency traffic lights per 100 grams in a recipe that includes these ingredients:",
        "Establish the Food Standards Agency traffic lights per 100 g in a recipe using these ingredients:",
        "Evaluate the FSA traffic light labels per 100 g in a recipe that has these ingredients:",
        "Analyze the Food Standards Agency traffic lights per 100 grams in a recipe with the following ingredients:",
        "Compute the Food Standards Agency traffic lights per 100 grams in a recipe that comprises these ingredients:",
        "Assess the fsa lights per 100 g in a recipe that includes the following ingredients:",
        "Review the fsa lights per 100 g in a recipe containing these ingredients:",
        "Check the Food Standards Agency traffic lights per 100 grams in a recipe using these ingredients:",
        "Verify the FSA traffic light labels per 100 grams in a recipe that has these ingredients:",
        "Gauge the Food Standards Agency traffic lights per 100 g in a recipe containing the following ingredients:",
        "Determine the fsa lights per 100 g in a recipe that comprises these ingredients:",
        "Calculate the Food Standards Agency traffic lights per 100 grams in a recipe with these ingredients:",
        "Find the fsa lights per 100 grams in a recipe using these ingredients:",
        "Identify the fsa lights per 100 g in a recipe that includes the following ingredients:",
        "Ascertain the fsa traffic lights per 100 g in a recipe with these ingredients:",
        "Discover the fsa lights per 100 grams in a recipe that contains the following ingredients:",
        "Establish the FSA traffic light labels per 100 grams in a recipe that includes these ingredients:",
        "Evaluate the fsa traffic lights per 100 g in a recipe that has these ingredients:",
        "Analyze the fsa lights per 100 g in a recipe with these ingredients:",
        "Compute the fsa traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Assess the Food Standards Agency traffic lights per 100 grams in a recipe with these ingredients:",
        "Review the fsa lights per 100 g in a recipe containing these ingredients:",
        "Check the fsa traffic lights per 100 g in a recipe that includes these ingredients:",
        "Verify the fsa traffic lights per 100 grams in a recipe that comprises these ingredients:",
        "Gauge the Food Standards Agency traffic lights per 100 grams in a recipe prepared with the following ingredients:",
        "Determine the Food Standards Agency traffic lights per 100 g in a recipe that uses the following ingredients:",
        "Calculate the fsa lights per 100 g in a recipe with these ingredients:",
        "Find the fsa traffic lights per 100 grams in a recipe containing these ingredients:",
        "Identify the Food Standards Agency traffic lights per 100 grams in a recipe that includes the following ingredients:",
        "Ascertain the Food Standards Agency traffic lights per 100 g in a recipe with these ingredients:",
        "Discover the fsa traffic lights per 100 g in a recipe prepared with these ingredients:",
        "Establish the fsa lights per 100 grams in a recipe that includes the following ingredients:",
        "Evaluate the Food Standards Agency traffic lights per 100 grams in a recipe with these ingredients:",
        "Analyze the Food Standards Agency traffic lights per 100 g in a recipe that uses these ingredients:",
        "Compute the FSA traffic light labels per 100 g in a recipe that comprises these ingredients:",
        "Assess the Food Standards Agency traffic lights per 100 grams in a recipe with the following ingredients:",
        "Review the fsa traffic lights per 100 grams in a recipe containing these ingredients:",
        "Check the fsa lights per 100 g in a recipe that includes these ingredients:",
        "Verify the fsa traffic lights per 100 g in a recipe prepared with these ingredients:",
        "Gauge the Food Standards Agency traffic lights per 100 g in a recipe that has the following ingredients:"
    ]
    fsa_lights_answer_prompts = [
        'The fsa lights per 100 g are:',
        'Fsa traffic lights per 100 g:',
        'Per 100 g, the Food Standards Agency traffic lights are:',
        'Per 100 grams, the Food Standards Agency traffic lights are:',
        'The fsa traffic lights for every 100 g are:',
        'Fsa lights for each 100 g:',
        'Each 100 g contains the following Food Standards Agency traffic lights:',
        'Food Standards Agency lights per 100 g:',
        'The Food Standards Agency lights per 100 g are:',
        'Food Standards Agency traffic lights in 100 g:',
        'FSA traffic light labels per 100 g:',
        'Values per 100 g of FSA traffic lights:',
        'Food Standards Agency traffic lights per 100 g:',
        'For every 100 g, the FSA traffic lights are:',
        'Each 100 grams has these Food Standards Agency traffic lights:',
        'Fsa traffic lights in 100 g:',
        'Food Standards Agency traffic lights in each 100 g:',
        'FSA traffic light labels per 100 g:',
        'Food Standards Agency traffic lights per 100 grams:',
        'The FSA traffic light labels per 100 g are:',
        'FSA lights per 100 g:',
        'FSA lights per 100 g:',
        'FSA traffic light labels per 100 g:',
        'The FSA traffic lights per 100 g are:',
        'Each 100 g provides these Food Standards Agency traffic lights:',
        'The Food Standards Agency traffic lights in 100 g are:',
        'In 100 g, the Food Standards Agency traffic lights are:',
        'For 100 g, the Food Standards Agency traffic lights are:',
        'FSA lights for every 100 g:',
        'Food Standards Agency lights in 100 g:',
        'Food Standards Agency traffic lights per 100 g listed:',
        'Every 100 g has these Food Standards Agency traffic lights:',
        'Per 100 g Food Standards Agency lights:',
        'FSA lights in 100 g:',
        'Food Standards Agency lights for each 100 g:',
        'The Food Standards Agency traffic lights found in 100 g are:',
        'In each 100 g, the Food Standards Agency traffic lights are:',
        'Food Standards Agency traffic lights in 100 g:',
        'FSA lights per 100 grams:',
        'FSA traffic light labels in 100 g:',
        'Values of FSA traffic lights per 100 g:',
        'The Food Standards Agency traffic lights for 100 g are:',
        'FSA traffic light labels per 100 grams:',
        'Food Standards Agency traffic lights in every 100 g:',
        'FSA traffic light labels in 100 g:',
        'Per 100 g, the FSA traffic lights are:',
        'FSA traffic light labels per 100 g:',
        'Food Standards Agency traffic lights in each 100 grams:',
        'The Food Standards Agency traffic lights per 100 grams are:',
        'Every 100 grams contains the following Food Standards Agency traffic lights:',
        'Fsa traffic lights per 100 grams:',
        'Food Standards Agency traffic lights in per 100 g:',
        'The fsa traffic lights for each 100 g are:',
        'For every 100 g, the Food Standards Agency traffic lights are:',
        'Each 100 g contains these Food Standards Agency traffic lights:',
        'Food Standards Agency traffic lights in 100 grams:',
        'Food Standards Agency lights in 100 g:',
        'FSA lights in 100 g:',
        'Per 100 g, the fsa lights are:',
        'The Food Standards Agency lights per 100 g are:',
        'FSA lights per 100 g:',
        'FSA lights per 100 grams:',
        'Fsa traffic lights in each 100 g:',
        'For 100 grams, the Food Standards Agency traffic lights are:',
        'FSA traffic light labels in 100 grams:',
        'Food Standards Agency traffic lights listed per 100 g:',
        'Each 100 grams provides these Food Standards Agency traffic lights:',
        'In every 100 g, the Food Standards Agency traffic lights are:',
        'FSA lights per 100 grams:',
        'The Food Standards Agency traffic lights in each 100 g are:',
        'Food Standards Agency traffic lights in every 100 grams:',
        'Values of FSA traffic lights in 100 g:',
        'FSA traffic light labels in 100 g:',
        'Each 100 g has these Food Standards Agency traffic lights:',
        'Food Standards Agency lights per 100 grams:',
        'Food Standards Agency traffic lights per 100 grams:',
        'FSA traffic light labels for every 100 g:',
        'Food Standards Agency lights per 100 g listed:',
        'The fsa traffic lights per 100 grams are:',
        'Fsa lights found in 100 g:',
        'For every 100 grams, the Food Standards Agency traffic lights are:',
        'Fsa lights in each 100 g:',
        'FSA lights in 100 grams:',
        'In 100 grams, the Food Standards Agency traffic lights are:',
        'Each 100 g provides the following Food Standards Agency traffic lights:',
        'Food Standards Agency traffic lights per 100 g given:',
        'Fsa lights listed per 100 g:',
        'The FSA traffic lights in 100 g are:',
        'Food Standards Agency lights for each 100 g:',
        'Food Standards Agency lights per 100 grams:',
        'Food Standards Agency traffic lights per 100 g presented:',
        'The Food Standards Agency traffic lights for each 100 g are:',
        'FSA traffic light labels in 100 g:',
        'Every 100 g has the following Food Standards Agency traffic lights:',
        'Fsa traffic lights for each 100 g:',
        'Per 100 grams Food Standards Agency lights:',
        'Food Standards Agency lights in each 100 g:',
        'FSA lights in 100 g:',
        'Values of FSA traffic lights per 100 grams:',
        'The Food Standards Agency traffic lights in 100 grams are:',
        'FSA lights for every 100 g:',
        'Food Standards Agency traffic lights for 100 g listed:',
        'Each 100 g contains the following Food Standards Agency traffic lights:',
        'FSA traffic light labels in each 100 g:',
        'The Food Standards Agency traffic lights per 100 g listed are:',
        'Food Standards Agency traffic lights per 100 grams given:',
        'For each 100 g, the Food Standards Agency traffic lights are:',
        'The fsa traffic lights per 100 g listed are:',
        'FSA lights in 100 g:',
        'Fsa lights per 100 g provided:',
        'Food Standards Agency traffic lights per 100 g detailed:',
        'Food Standards Agency traffic lights listed in 100 g:',
        'The FSA traffic lights for each 100 g are:',
        'Food Standards Agency traffic lights per 100 g found:',
        'Per 100 grams, the fsa lights are:',
        'FSA traffic light labels for 100 grams:',
        'Food Standards Agency traffic lights per 100 g shown:',
        'Fsa lights for each 100 grams:',
        'Each 100 grams has these Food Standards Agency traffic lights:',
        'Food Standards Agency lights for every 100 g:',
        'Food Standards Agency traffic lights per 100 grams listed:',
        'Fsa lights for 100 g are:',
        'Food Standards Agency traffic lights for each 100 grams:',
        'Fsa traffic lights per 100 g detailed:',
        'The FSA lights per 100 g are:',
        'Food Standards Agency traffic lights in each 100 grams are:',
        'Per 100 grams the Food Standards Agency traffic lights are:',
        'Food Standards Agency traffic lights per 100 g provided:',
        'Fsa lights given per 100 g:',
        'Food Standards Agency traffic lights for 100 grams:',
        'Food Standards Agency lights detailed per 100 g:',
        'The Food Standards Agency traffic lights in every 100 g are:',
        'Food Standards Agency traffic lights per 100 g highlighted:',
        'Food Standards Agency traffic lights detailed for 100 g:',
        'Fsa lights in every 100 g:',
        'Food Standards Agency traffic lights detailed per 100 grams:',
        'Each 100 g has these fsa lights:',
        'Food Standards Agency traffic lights shown for 100 g:',
        'Food Standards Agency traffic lights highlighted per 100 g:',
        'The Food Standards Agency traffic lights detailed for 100 g are:',
        'Food Standards Agency traffic lights highlighted for 100 grams:',
        'The Food Standards Agency traffic lights provided per 100 g are:',
        'Food Standards Agency traffic lights provided per 100 grams:',
        'Food Standards Agency traffic lights per 100 g listed above:',
        'The Food Standards Agency traffic lights per 100 grams given are:',
        'Food Standards Agency traffic lights per 100 grams shown:',
        'Each 100 grams provides the following Food Standards Agency traffic lights:',
        'The Food Standards Agency traffic lights highlighted per 100 g are:'
    ]

    def generate(self, sample: list, ingredients: list) -> dict:
        questions = []
        answers = []

        for i, recipe, _, _ in sample:
            question = ''

            for ingredient, quantity, unit in zip(recipe.ingredients, recipe.ingredient_quantity,
                                                  recipe.ingredient_unit):
                question += f'{quantity} {unit} {ingredient}, '
            question = question[:-2]

            answer = (f'fat - {recipe.fat_fsa}, '
                      f'salt - {recipe.salt_fsa}, saturates - {recipe.saturates_fsa}, sugars - {recipe.sugars_fsa}')

            question_prompt = self.fsa_lights_question_prompts[
                self.generate_random_number(0, len(self.fsa_lights_question_prompts) - 1)]

            questions.append(f'{self.start} '
                             f'{question_prompt.replace('a recipe', f'{recipe.title}')}'
                             f' {question.strip()} {self.end}')
            answers.append(
                f'{self.fsa_lights_answer_prompts[
                    self.generate_random_number(0, len(self.fsa_lights_answer_prompts) - 1)]}'
                f' {answer}')

        return {'questions': questions, 'answers': answers}


if __name__ == '__main__':
    max_count = 20
    max_jaccard = 0.75
    random_seed = 42  # replace with appropriate random seed

    file = r'path\to\input\json\file'
    directory = r'path\to\output\directory'

    writer = Writer()

    # recipe ingredients -> nutritional values
    generator = RecipeIngrNutritionSequenceGenerator(random_seed=random_seed,
                                                     max_count=max_count,
                                                     max_jaccard=max_jaccard)
    sequence, sequence_test = generator.generate_sequences(file)

    writer.write(
        save_file=os.path.join(directory, 'recipes_ing_nutrition', str(random_seed),
                               f'dataset_training_{generator.max_count}_{random_seed}.txt'),
        questions=sequence['questions'],
        answers=sequence['answers'],
        append=False
    )

    writer.write(
        save_file=os.path.join(directory, 'recipes_ing_nutrition', str(random_seed),
                               f'dataset_test_{generator.max_count}_{random_seed}.txt'),
        questions=sequence_test['questions'],
        answers=sequence_test['answers'],
        append=False
    )

    # recipe title and ingredients -> nutritional values
    generator = RecipeTitlIngrNutritionSequenceGenerator(random_seed=random_seed,
                                                         max_count=max_count,
                                                         max_jaccard=max_jaccard)
    sequence, sequence_test = generator.generate_sequences(file)

    writer.write(
        save_file=os.path.join(directory, 'recipes_title_ing_nutrition', str(random_seed),
                               f'dataset_training_{generator.max_count}_{random_seed}.txt'),
        questions=sequence['questions'],
        answers=sequence['answers'],
        append=False
    )

    writer.write(
        save_file=os.path.join(directory, 'recipes_title_ing_nutrition', str(random_seed),
                               f'dataset_test_{generator.max_count}_{random_seed}.txt'),
        questions=sequence_test['questions'],
        answers=sequence_test['answers'],
        append=False
    )

    # recipe ingredients -> fsa lights
    generator = RecipeIngrFsaSequenceGenerator(random_seed=random_seed,
                                               max_count=max_count,
                                               max_jaccard=max_jaccard)
    sequence, sequence_test = generator.generate_sequences(file)

    writer.write(
        save_file=os.path.join(directory, 'recipes_ing_fsa', str(random_seed),
                               f'dataset_training_{generator.max_count}_{random_seed}.txt'),
        questions=sequence['questions'],
        answers=sequence['answers'],
        append=False
    )

    writer.write(
        save_file=os.path.join(directory, 'recipes_ing_fsa', str(random_seed),
                               f'dataset_test_{generator.max_count}_{random_seed}.txt'),
        questions=sequence_test['questions'],
        answers=sequence_test['answers'],
        append=False
    )

    # recipe title and ingredients -> fsa lights
    generator = RecipeTitlIngrFsaSequenceGenerator(random_seed=random_seed,
                                                   max_count=max_count,
                                                   max_jaccard=max_jaccard)
    sequence, sequence_test = generator.generate_sequences(file)

    writer.write(
        save_file=os.path.join(directory, 'recipes_title_ing_fsa', str(random_seed),
                               f'dataset_training_{generator.max_count}_{random_seed}.txt'),
        questions=sequence['questions'],
        answers=sequence['answers'],
        append=False
    )

    writer.write(
        save_file=os.path.join(directory, 'recipes_title_ing_fsa', str(random_seed),
                               f'dataset_test_{generator.max_count}_{random_seed}.txt'),
        questions=sequence_test['questions'],
        answers=sequence_test['answers'],
        append=False
    )
