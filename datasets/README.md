# Fine-Tuning Datasets

This folder contains the code used to generate the LLM fine-tuning datasets. It allows generation of fine-tuning datasets for the following tasks:

- **`Assessing ingredient nutritional profile`**
- **`Assessing recipe nutritional profile`**
- **`Classifying recipes by traffic lights nutrition labels`**

To generate the datasets, run the following scripts, after obtaining the Recipe1M+ dataset and configuring the scripts with the required parameters:

- **`nutrients_fsa_lights/IngredientSampling.py`**: Outputs the training and test datasets for task **`Assessing ingredient nutritional profile`**. Configuration parameters: (1) input file path and (2) output directory path.
- **`nutrients_fsa_lights/RecipeSampling.py`**: Outputs the training and test datasets for tasks **`Assessing recipe nutritional profile`** and **`Classifying recipes by traffic lights nutrition labels`**, for two variations of the input prompt (containing only the recipe ingredient list or containing the recipe title concatenated with the ingredient list). Configuration parameters: (1) input file path, (2) output directory path, and (3) random seed. Run it five times for each task and input prompt variation, using the appropriate random seeds listed below.

Use the following random seeds for the appropriate tasks in script **`nutrients_fsa_lights/RecipeSampling.py`**:

- **`Assessing recipe nutritional profile (ingredient list)`**: 107473, 234053, 442417, 619176, 777572
- **`Assessing recipe nutritional profile (recipe title and ingredient list)`**: 206184, 384578, 512894, 638165, 767065
- **`Classifying recipes by traffic lights nutrition labels (ingredient list)`**: 120459, 228727, 398065, 537363, 638908
- **`Classifying recipes by traffic lights nutrition labels (recipe title and ingredient list)`**: 150769, 359225, 476390, 755236, 812461
