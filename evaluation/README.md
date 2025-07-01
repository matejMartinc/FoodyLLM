# Evaluation

This folder contains the code used to evaluate FoodyLLM and is organised as follows:

- **`entities.py`**: Entities specific to each evaluation task, implementing the core evaluation logic (e.g., label modification, performance score calculation by instance and dataset).
- **`modifiers.py`**: Modifiers of the input data to bring it to standardized format (as applicable). Use **`base`** modifiers to process FoodyLLM output and **`extended`** or **`synonym`** modifiers to process baseline (non-fine-tuned) LLM output.
- **`pipelines.py`**: Evaluation pipelines to be run for each specific task.

The tab-separated input files should contain the following columns:

- **`Id`**: Identifier of the instance.
- **`Original prompt`**: Original prompt in the dataset.
- **`True prompt`**: Modified prompt used in evaluation.
- **`Answer`**: Answer outputted by the LLM.
- **`True`**: True answer.
