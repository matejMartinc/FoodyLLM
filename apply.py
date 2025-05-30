from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

if __name__ == '__main__':
    base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    tokenizer.pad_token = '<|pad|>'
    tokenizer.pad_token_id = 128255

    #Load LORA weights
    model.load_adapter("Matej/FoodyLLM")
    model.config.use_cache = True
    model.eval()

    #Return the nutrient values for an example recipe
    system_prompt = ""
    user_prompt = "Compute the nutrient values per 100 grams in a recipe with the following ingredients: 250 g cream, whipped, cream topping, pressurized, 250 g yogurt, greek, plain, nonfat, 50 g sugars, powdered"

    messages = [
        {
            "role": "user",
            "content": f"{system_prompt} {user_prompt}".strip()
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    #Here we have a batch of one
    tokenizer_input = [prompt]

    inputs = tokenizer(tokenizer_input, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True)
    answers = tokenizer.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:])
    answers = [x.split('<|eot_id|>')[0].strip() for x in answers]
    print(answers[0])
    #Expected answer: Nutrient values per 100 g highlighted: energy - 134.24, fat - 5.78, protein - 7.51, salt - 0.06, saturates - 3.58, sugars - 13.00


    #Classifying recipes by traffic light nutrition labels
    user_prompt = "Review the fsa traffic lights per 100 grams in a recipe using the following ingredients: 1/2 cup soup, swanson chicken broth 99% fat free, 1 pinch salt, table"

    messages = [
        {
            "role": "user",
            "content": f"{system_prompt} {user_prompt}".strip()
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Here we have a batch of one
    tokenizer_input = [prompt]

    inputs = tokenizer(tokenizer_input, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True)
    answers = tokenizer.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:])
    answers = [x.split('<|eot_id|>')[0].strip() for x in answers]
    print(answers[0])
    #Expected answer: Food Standards Agency lights in each 100 g: fat - green, salt - red, saturates - green, sugars - green


    #Extract food named entities
    user_prompt = "Retrieve all food entities referenced in the text: Line a large colander with a cheesecloth. Stir salt into the yogurt, and pour the yogurt into the cheesecloth. Set the colander in the sink or bowl to catch the liquid that drains off. Leave to drain for 24 hours. After draining for the 24 hours, transfer the resulting cheese to a bowl. Stir in the olive oil. Store in a covered container in the refrigerator."
    messages = [
        {
            "role": "user",
            "content": f"{system_prompt} {user_prompt}".strip()
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Here we have a batch of one
    tokenizer_input = [prompt]

    inputs = tokenizer(tokenizer_input, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True)
    answers = tokenizer.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:])
    answers = [x.split('<|eot_id|>')[0].strip() for x in answers]
    print(answers[0])
    #Expected answer: Indeed, the entities concerning food are outlined below:  salt, yogurt, liquid, cheese, olive oil.

    #Link named entities to the SNOMEDCT ontology
    user_prompt = "Link the following food entities to a SNOMEDCT ontology: cream cheese, meat"
    messages = [
        {
            "role": "user",
            "content": f"{system_prompt} {user_prompt}".strip()
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Here we have a batch of one
    tokenizer_input = [prompt]

    inputs = tokenizer(tokenizer_input, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True)
    answers = tokenizer.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:])
    answers = [x.split('<|eot_id|>')[0].strip() for x in answers]
    print(answers[0])
    #Expected answer: Indeed, the entities are connected in this fashion:  cream cheese - http://purl.bioontology.org/ontology/SNOMEDCT/226849005;http://purl.bioontology.org/ontology/SNOMEDCT/255621006;http://purl.bioontology.org/ontology/SNOMEDCT/102264005, meat - http://purl.bioontology.org/ontology/SNOMEDCT/28647000.
