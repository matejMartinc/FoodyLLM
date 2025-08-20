class Recipe:
    def __init__(self, raw_info: dict):
        self.id_ = self.parse_attribute(raw_info, 'id')
        self.partition = self.parse_attribute(raw_info, 'partition')
        self.title = self.parse_attribute(raw_info, 'title')

        [self.fat_fsa,
         self.salt_fsa,
         self.saturates_fsa,
         self.sugars_fsa] = self.parse_fsa_lights_per100g(raw_info)

        [self.energy_value,
         self.fat_value,
         self.protein_value,
         self.salt_value,
         self.saturates_value,
         self.sugars_value] = self.parse_nutr_values_per100g(raw_info)

        self.ingredients = self.parse_dictionary(raw_info, 'ingredients')
        self.ingredient_quantity = self.parse_dictionary(raw_info, 'quantity')
        self.ingredient_unit = self.parse_dictionary(raw_info, 'unit')
        self.ingredient_weight = self.parse_attribute(raw_info, 'weight_per_ingr')

        self.instructions = self.parse_dictionary(raw_info, 'instructions')

        [self.ingredient_fat,
         self.ingredient_nrg,
         self.ingredient_pro,
         self.ingredient_sat,
         self.ingredient_sod,
         self.ingredient_sug] = self.parse_nutr_per_ingredient(raw_info)

    @staticmethod
    def parse_attribute(raw_info, attribute):
        return raw_info[attribute]

    @staticmethod
    def parse_dictionary(raw_info, element):
        ingredients = []
        for ingredient in raw_info[element]:
            ingredients.append(ingredient['text'])
        return ingredients

    @staticmethod
    def parse_nutr_values_per100g(raw_info):
        values = raw_info['nutr_values_per100g']
        return [values['energy'],
                values['fat'],
                values['protein'],
                values['salt'],
                values['saturates'],
                values['sugars']]

    @staticmethod
    def parse_nutr_per_ingredient(raw_info):
        fat = []
        nrg = []
        pro = []
        sat = []
        sod = []
        sug = []

        for ingredient in raw_info['nutr_per_ingredient']:
            fat.append(ingredient['fat'])
            nrg.append(ingredient['nrg'])
            pro.append(ingredient['pro'])
            sat.append(ingredient['sat'])
            sod.append(ingredient['sod'])
            sug.append(ingredient['sug'])
        return [fat, nrg, pro, sat, sod, sug]

    @staticmethod
    def parse_fsa_lights_per100g(raw_info):
        values = raw_info['fsa_lights_per100g']
        return [values['fat'],
                values['salt'],
                values['saturates'],
                values['sugars']]
