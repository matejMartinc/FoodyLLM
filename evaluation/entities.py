import re
from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score


class Evaluator:
    @staticmethod
    def f1(y_true: list, y_pred: list, labels: list, average: str, round_=True):
        p = precision_score(y_true=y_true, y_pred=y_pred, average=average, labels=labels)
        r = recall_score(y_true=y_true, y_pred=y_pred, average=average, labels=labels)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average, labels=labels)
        if round_:
            return [f'{p:.3f}', f'{r:.3f}', f'{f1:.3f}']
        else:
            return [p, r, f1]


class NutrientAnnotator(ABC):
    @staticmethod
    @abstractmethod
    def annotate(predicted_value: float, true_value: float, method_a: bool = True):
        raise NotImplementedError()


class FatAnnotator(NutrientAnnotator):
    @staticmethod
    def annotate(predicted_value: float, true_value: float, method_a: bool = True):
        difference = abs(predicted_value - true_value)

        if true_value < 10.0:
            if difference <= 1.5:
                correct = 1
            else:
                correct = 0
        elif (true_value >= 10.0) and (true_value <= 40.0):
            if difference <= (0.2 * true_value):
                correct = 1
            else:
                correct = 0
        elif true_value > 40.0:
            if difference <= 8.0:
                correct = 1
            else:
                correct = 0

        return correct


class ProteinSugarAnnotator(NutrientAnnotator):
    @staticmethod
    def annotate(predicted_value: float, true_value: float, method_a: bool = True):
        difference = abs(predicted_value - true_value)

        if true_value < 10.0:
            if difference <= 2.0:
                correct = 1
            else:
                correct = 0
        elif (true_value >= 10.0) and (true_value <= 40.0):
            if difference <= (0.2 * true_value):
                correct = 1
            else:
                correct = 0
        elif true_value > 40.0:
            if difference <= 8.0:
                correct = 1
            else:
                correct = 0

        return correct


class SaturatesAnnotator(NutrientAnnotator):
    @staticmethod
    def annotate(predicted_value: float, true_value: float, method_a: bool = True):
        difference = abs(predicted_value - true_value)

        if true_value < 4.0:
            if difference <= 0.8:
                correct = 1
            else:
                correct = 0
        elif true_value >= 4.0:
            if difference <= (0.2 * true_value):
                correct = 1
            else:
                correct = 0

        return correct


class SaltAnnotator(NutrientAnnotator):
    @staticmethod
    def annotate(predicted_value: float, true_value: float, method_a: bool = True):
        difference = abs(predicted_value - true_value)

        if true_value < 1.25:
            if difference <= 0.375:
                correct = 1
            else:
                correct = 0
        elif true_value >= 1.25:
            if difference <= (0.2 * true_value):
                correct = 1
            else:
                correct = 0

        return correct


class Concept(ABC):
    def __init__(self, id_: str, concept: str,
                 true_entities_str: str, predicted_entities_str: str, delimiter: str):
        self.id_ = id_
        self.concept = concept
        self.delimiter = delimiter

        self.note = ' '

        self.true_label_vector = None
        self.predicted_label_vector = None

        [self.valid_true_concept, self.true_entities] = self.parse_true_entities(
            true_entities_str, delimiter)
        [self.valid_predicted_concept, self.predicted_entities] = self.parse_predicted_entities(
            predicted_entities_str, delimiter)

        original_true_entities = self.true_entities.copy()
        original_predicted_entities = self.predicted_entities.copy()

        print(predicted_entities_str)

        if self.valid_true_concept:
            self.true_entities = self.modify_true_entities(original_true_entities, original_predicted_entities)

        self.predicted_entities = self.modify_predicted_entities(original_true_entities,
                                                                 original_predicted_entities)

    def parse_true_entities(self, text: str, delimiter: str):
        try:
            entities = text.split(delimiter)
            entities = [entity[:-1].strip() if entity.endswith('.') else entity.strip() for entity in entities]
        except Exception as e:
            print(e)
            self.note += f'Invalid true answer entities for concept {self.concept}: {text}. '
            return [False, []]

        return [True, entities]

    def parse_predicted_entities(self, text: str, delimiter: str):
        try:
            entities = text.split(delimiter)
            entities = [entity[:-1].strip() if entity.endswith('.') else entity.strip() for entity in entities]
        except Exception as e:
            print(e)
            self.note += f'Invalid predicted answer entities for concept {self.concept}: {text}. '
            return [False, []]

        return [True, entities]

    @abstractmethod
    def modify_true_entities(self, true_entities: list, predicted_entities: list):
        raise NotImplementedError()

    @abstractmethod
    def modify_predicted_entities(self, true_entities: list, predicted_entities: list):
        raise NotImplementedError()

    def set_label_vectors(self, true_label_vector: list, predicted_label_vector: list):
        self.true_label_vector = true_label_vector
        self.predicted_label_vector = predicted_label_vector

    def print(self):
        return [[self.id_, ' ', ' ', ' ', ' ', ' ', self.concept,
                 '\n'.join(sorted(self.true_entities)),
                 '\n'.join(sorted(self.predicted_entities)), self.note, ' ', ' ', ' '],
                self.true_label_vector,
                self.predicted_label_vector]


class NelConcept(Concept):
    def modify_true_entities(self, true_entities: list, predicted_entities: list):
        return true_entities

    def modify_predicted_entities(self, true_entities: list, predicted_entities: list):
        return predicted_entities


class FsaConcept(Concept):
    def modify_true_entities(self, true_entities: list, predicted_entities: list):
        return true_entities

    def modify_predicted_entities(self, true_entities: list, predicted_entities: list):
        return predicted_entities


class NutritionConcept(Concept):
    def modify_true_entities(self, true_entities: list, predicted_entities: list):
        modified_true_entities = []
        for entity in true_entities:
            if 'energy' in entity:
                continue
            modified_entity = re.sub(' - [0-9]+.[0-9]{2}', '', entity)

            print(f'old: {entity}\tnew: {modified_entity}')
            modified_true_entities.append(modified_entity)
        return modified_true_entities

    def modify_predicted_entities(self, true_entities: list, predicted_entities: list):
        prefixes = ['fat - ', 'protein - ', 'salt - ', 'saturates - ', 'sugars - ']
        labels = ['fat', 'protein', 'salt', 'saturates', 'sugars']
        annotators = [FatAnnotator, ProteinSugarAnnotator, SaltAnnotator, SaturatesAnnotator, ProteinSugarAnnotator]

        modified_predicted_entities = []
        for true_entity, predicted_entity in zip(true_entities, predicted_entities):
            for prefix, label, annotator in zip(prefixes, labels, annotators):
                if prefix in true_entity and prefix in predicted_entity:
                    print(predicted_entity)
                    true_value = float(true_entity.strip().replace(prefix, ''))

                    predicted_entity = predicted_entity.replace('"', '').strip()
                    predicted_entity = predicted_entity[:-1] if predicted_entity.endswith('.') else predicted_entity
                    print(predicted_entity)
                    predicted_value = float(predicted_entity.replace('"', '').strip().replace(prefix, ''))

                    modified_predicted_value = annotator.annotate(
                        predicted_value=predicted_value, true_value=true_value)
                    print(f'old: {predicted_value}\tnew: {modified_predicted_value}')
                    if modified_predicted_value == 1:
                        modified_predicted_entities.append(f'{label}')

        return modified_predicted_entities


class Instance(ABC):
    def __init__(self, id_: int, original_prompt: str, true_prompt: str,
                 predicted_answer: str, actual_predicted_answer: str, true_answer: str):
        self.id_ = str(id_)

        self.original_prompt = original_prompt
        self.true_prompt = true_prompt
        self.predicted_answer = predicted_answer
        self.true_answer = true_answer
        self.actual_predicted_answer = actual_predicted_answer

        self.note = ' '
        self.concept_counter = 0

        [self.valid_original_prompt_concepts, self.original_prompt_concepts] = self.parse_concepts(
            self.original_prompt)
        [self.valid_true_answer, self.true_concepts] = self.parse_true_answer_concepts(
            self.true_answer, self.original_prompt_concepts)
        [self.valid_predicted_answer, self.predicted_concepts] = self.parse_predicted_answer_concepts(
            self.predicted_answer, self.true_concepts)

        self.concepts = []
        if self.valid_true_answer:
            for key in self.true_concepts:
                self.concepts.append(self.create_concept(
                    id_=self.id_ + '_' + str(self.concept_counter),
                    concept=key,
                    true_entities=self.true_concepts,
                    predicted_entities=self.predicted_concepts))
                self.concept_counter += 1
        else:
            self.note += f'Invalid true answer: {self.true_answer}. '

    @abstractmethod
    def create_concept(self, instance_id: int, concept: str, true_entities: list, predicted_entities: list):
        raise NotImplementedError()

    @abstractmethod
    def parse_concepts(self, text: str):
        raise NotImplementedError()

    @abstractmethod
    def parse_true_answer_concepts(self, text: str, original_prompt_concepts: list):
        raise NotImplementedError()

    @abstractmethod
    def parse_predicted_answer_concepts(self, text: str, true_concepts: list):
        raise NotImplementedError()

    @abstractmethod
    def print(self, labels: list):
        raise NotImplementedError()


class NelInstance(Instance):
    def create_concept(self, id_: str, concept: str, true_entities: list, predicted_entities: list):
        new_concept = NelConcept(
            id_=id_,
            concept=concept,
            true_entities_str=true_entities[concept],
            predicted_entities_str=(predicted_entities[concept]
                                    if concept in predicted_entities.keys() else ''),
            delimiter=';')
        return new_concept

    def parse_concepts(self, text: str):
        try:
            parts = text.split(': ')
            parts = [p.strip() for p in parts]

            concepts = parts[1].split(',')
            concepts = [concept[:-1].strip() if concept.endswith('?') else concept.strip() for concept in concepts]
            concepts = [concept.strip() for concept in concepts if concept != '']
        except Exception as e:
            print(e)
            self.note += f'Invalid prompt concepts: {text}. '
            return [False, []]

        return [True, concepts]

    def parse_true_answer_concepts(self, text: str, original_prompt_concepts: list):
        try:
            concepts = dict()
            parts = text.split(': ')
            parts = [p.strip() for p in parts]

            parts = [re.sub(r'\s\[[A-Za-z\-,/\\\s()]+]\.?', '', p) for p in parts]

            concept_entities = parts[1].split(',')
            print(concept_entities)

            corrections_made = True
            while corrections_made:
                corrections_made = False
                new_concept_entities = []
                i = 0
                while i < len(concept_entities):
                    print(i)
                    concept_entity = concept_entities[i]
                    if (' - ' not in concept_entity) and (i + 1 < len(concept_entities)):
                        concept_entity = concept_entity + ',' + concept_entities[i + 1]
                        new_concept_entities.append(concept_entity)
                        corrections_made = True
                        i += 2
                    else:
                        new_concept_entities.append(concept_entity)
                        i += 1
                concept_entities = new_concept_entities.copy()
                print(concept_entities)

            print(new_concept_entities)
            new_concept_entities = [concept[:-1].strip() if concept.endswith('.') else
                                    concept.strip() for concept in new_concept_entities]
            new_concept_entities = [concept.strip() for concept in new_concept_entities if concept != '']

            print(new_concept_entities)
            for concept_entity in new_concept_entities:
                [concept, entity_str] = concept_entity.split(' - ')
                concept = concept.strip().lower()
                concepts[concept] = (entity_str[:-1].strip() if entity_str.endswith('.') else entity_str.strip())

        except Exception as e:
            print(e)
            self.note += f'Invalid true answer concepts: {text}. '
            return [False, dict()]

        return [True, concepts]

    def parse_predicted_answer_concepts(self, text: str, true_concepts: list):
        try:
            concepts = dict()
            parts = text.split(': ')
            parts = [p.strip() for p in parts]
            parts = [re.sub(r'\s\[[A-Za-z\-,/\\\s()]+]\.?', '', p) for p in parts]
            assert len(parts) == 2

            concept_entities = parts[1].split(',')
            for concept_entity in concept_entities:
                if concept_entity == '':
                    continue
                if (' - ' not in concept_entity) and (concept_entity.strip().lower() in true_concepts):
                    [concept, entity_str] = [concept_entity, '']
                elif (' - ' not in concept_entity) and (concept_entity.startswith('age')):
                    self.note += f'Invalid predicted answer concept without entities: {concept_entity}. '
                    print(concept_entity)
                    [concept, entity_str] = [concept_entity, '']
                else:
                    [concept, entity_str] = concept_entity.split(' - ')
                concept = concept.strip().lower()
                concepts[concept] = (entity_str[:-1].strip() if entity_str.endswith('.') else entity_str.strip())
        except Exception as e:
            print(e)
            self.note += f'Invalid predicted answer concepts: {text}. {e} '
            return [False, dict()]

        return [True, concepts]

    def print(self, labels: list):
        concept_rows = []
        instance_y_true_matrix = None
        instance_y_pred_matrix = None
        for concept in self.concepts:
            [rows, y_true_matrix, y_pred_matrix] = concept.print()
            concept_rows = concept_rows + [rows]
            if instance_y_true_matrix is None:
                instance_y_true_matrix = y_true_matrix
                instance_y_pred_matrix = y_pred_matrix
            else:
                instance_y_true_matrix = np.concatenate([instance_y_true_matrix, y_true_matrix])
                instance_y_pred_matrix = np.concatenate([instance_y_pred_matrix, y_pred_matrix])

        if instance_y_true_matrix is not None and instance_y_pred_matrix is not None:
            instance_result_row = (
                    [self.id_, self.original_prompt, self.true_answer,
                     self.actual_predicted_answer, self.predicted_answer,
                     ' ', ' ', ' ', ' ', ' '] +
                    Evaluator.f1(
                        y_true=instance_y_true_matrix,
                        y_pred=instance_y_pred_matrix,
                        average='weighted', labels=labels)
            )
        else:
            instance_result_row = (
                    [self.id_, self.original_prompt, self.true_answer,
                     self.actual_predicted_answer, self.predicted_answer, self.note,
                     ' ', ' ', ' ', ' '] +
                    [None, None, None]
            )
        return [[instance_result_row] + concept_rows, instance_y_true_matrix, instance_y_pred_matrix]


class FsaInstance(Instance):
    def create_concept(self, id_: str, concept: str, true_entities: list, predicted_entities: list):
        new_concept = FsaConcept(
            id_=id_,
            concept=concept,
            true_entities_str=true_entities[concept],
            predicted_entities_str=(predicted_entities[concept]
                                    if concept in predicted_entities.keys() else ''),
            delimiter=', ')
        return new_concept

    def parse_concepts(self, text: str):
        concepts = ['concept']
        return [True, concepts]

    def parse_true_answer_concepts(self, text: str, original_prompt_concepts: list):
        try:
            concepts = dict()
            parts = text.split(': ')
            parts = [p.strip() for p in parts]

            concepts['concept'] = parts[1].strip().lower()
        except Exception as e:
            print(e)
            self.note += f'Invalid true answer concepts: {text}. '
            return [False, dict()]

        return [True, concepts]

    def parse_predicted_answer_concepts(self, text: str, true_concepts: list):
        try:
            concepts = dict()
            parts = text.split(': ')
            parts = [p.strip() for p in parts]

            concepts['concept'] = parts[1].strip().lower()
        except Exception as e:
            print(e)
            self.note += f'Invalid predicted answer concepts: {text}. '
            return [False, dict()]

        return [True, concepts]

    def print(self, labels: list):
        instance_y_true_matrix = None
        instance_y_pred_matrix = None

        assert len(self.concepts) == 1
        concept = self.concepts[0]
        [row, y_true_matrix, y_pred_matrix] = concept.print()

        if instance_y_true_matrix is None:
            instance_y_true_matrix = y_true_matrix
            instance_y_pred_matrix = y_pred_matrix
        else:
            instance_y_true_matrix = np.concatenate([instance_y_true_matrix, y_true_matrix])
            instance_y_pred_matrix = np.concatenate([instance_y_pred_matrix, y_pred_matrix])

        if instance_y_true_matrix is not None and instance_y_pred_matrix is not None:
            instance_result_row = (
                    [self.id_, self.original_prompt, self.true_answer, self.actual_predicted_answer, self.predicted_answer, self.note,
                     row[6], row[7], row[8], row[9]] +
                    Evaluator.f1(
                        y_true=instance_y_true_matrix,
                        y_pred=instance_y_pred_matrix,
                        average='weighted', labels=labels)
            )
        else:
            instance_result_row = (
                    [self.id_, self.original_prompt, self.true_answer, self.actual_predicted_answer, self.predicted_answer, self.note,
                     row[6], row[7], row[8], row[9]] +
                    [None, None, None]
            )
        return [[instance_result_row], instance_y_true_matrix, instance_y_pred_matrix]


class NutritionInstance(Instance):
    def create_concept(self, id_: str, concept: str, true_entities: list, predicted_entities: list):
        new_concept = NutritionConcept(
            id_=id_,
            concept=concept,
            true_entities_str=true_entities[concept],
            predicted_entities_str=(predicted_entities[concept]
                                    if concept in predicted_entities.keys() else ''),
            delimiter=', ')
        return new_concept

    def parse_concepts(self, text: str):
        concepts = ['concept']
        return [True, concepts]

    def parse_true_answer_concepts(self, text: str, original_prompt_concepts: list):
        try:
            concepts = dict()
            parts = text.split(': ')
            parts = [p.strip() for p in parts]

            concepts['concept'] = parts[1].strip().lower()
        except Exception as e:
            print(e)
            self.note += f'Invalid true answer concepts: {text}. '
            return [False, dict()]

        return [True, concepts]

    def parse_predicted_answer_concepts(self, text: str, true_concepts: list):
        try:
            concepts = dict()
            parts = text.split(': ')
            parts = [p.strip() for p in parts]

            concepts['concept'] = parts[1].strip().lower()
        except Exception as e:
            print(e)
            self.note += f'Invalid predicted answer concepts: {text}. '
            return [False, dict()]

        return [True, concepts]

    def print(self, labels: list):
        concept_rows = []
        instance_y_true_matrix = None
        instance_y_pred_matrix = None

        assert len(self.concepts) == 1
        concept = self.concepts[0]
        [row, y_true_matrix, y_pred_matrix] = concept.print()

        if instance_y_true_matrix is None:
            instance_y_true_matrix = y_true_matrix
            instance_y_pred_matrix = y_pred_matrix
        else:
            instance_y_true_matrix = np.concatenate([instance_y_true_matrix, y_true_matrix])
            instance_y_pred_matrix = np.concatenate([instance_y_pred_matrix, y_pred_matrix])

        if instance_y_true_matrix is not None and instance_y_pred_matrix is not None:
            instance_result_row = (
                    [self.id_, self.original_prompt, self.true_answer, self.actual_predicted_answer, self.predicted_answer, self.note,
                     row[6], row[7], row[8], row[9]] +
                    Evaluator.f1(
                        y_true=instance_y_true_matrix,
                        y_pred=instance_y_pred_matrix,
                        average='weighted', labels=labels)
            )
        else:
            instance_result_row = (
                    [self.id_, self.original_prompt, self.true_answer, self.actual_predicted_answer, self.predicted_answer, self.note,
                     row[6], row[7], row[8], row[9]] +
                    [None, None, None]
            )
        return [[instance_result_row], instance_y_true_matrix, instance_y_pred_matrix]


class Dataset:
    def __init__(self):
        self.all_instance_counter = 0
        self.test_instance_counter = 0

        self.all_instances = []
        self.all_concept_entities = []
        self.test_instances = []

        self.label_transformer = MultiLabelBinarizer()

    @abstractmethod
    def add_instance(self, id_: int, original_prompt: str, true_prompt: str, true_answer: str,
                     predicted_answer: str, actual_predicted_answer: str):
        raise NotImplementedError()

    def add_labels(self, prompt: str, answer: str):
        instance = self.add_instance(
            id_=self.all_instance_counter,
            original_prompt=prompt,
            true_prompt=prompt,
            true_answer=answer,
            predicted_answer=answer,
            actual_predicted_answer=answer)
        if not instance.valid_true_answer:
            print(instance.true_answer)
            print(instance.note)
            print()

        for concept in instance.concepts:
            self.all_concept_entities = self.all_concept_entities + [concept.true_entities]

        self.all_instances.append(instance)
        self.all_instance_counter += 1

    def add_test_instance(self, original_prompt: str, true_prompt: str,
                          predicted_answer: str, actual_predicted_answer:str, true_answer: str):
        instance = self.add_instance(
            id_=self.test_instance_counter,
            original_prompt=original_prompt,
            true_prompt=true_prompt,
            true_answer=true_answer,
            predicted_answer=predicted_answer,
            actual_predicted_answer=actual_predicted_answer
        )

        self.test_instances.append(instance)
        self.test_instance_counter += 1

    def fit_labels(self):
        self.label_transformer = self.label_transformer.fit(self.all_concept_entities)
        for instance in self.test_instances:
            if instance.valid_original_prompt_concepts and instance.valid_true_answer:
                for concept in instance.concepts:
                    true_label_vector = self._transform_labels_to_vector(concept.true_entities)
                    predicted_label_vector = self._transform_labels_to_vector(concept.predicted_entities)
                    concept.set_label_vectors(true_label_vector=true_label_vector,
                                              predicted_label_vector=predicted_label_vector)

    def _transform_labels_to_vector(self, labels: list):
        return self.label_transformer.transform([labels])

    def print(self):
        header = ['instance_id', 'original_prompt', 'true_answer', 'actual_predicted_answer', 'predicted_answer', 'instance_notes',
                  'concept', 'true_entities', 'predicted_entities', 'concept_notes',
                  'macro_weighted_precision', 'macro_weighted_recall', 'macro_weighted_f1']

        instance_rows = []
        dataset_y_true_matrix = None
        dataset_y_pred_matrix = None

        label_ids = range(len(self.label_transformer.classes_))
        labels = self.label_transformer.classes_

        total = len(self.test_instances)
        skipped = 0
        used = 0
        for instance in self.test_instances:
            if instance.valid_original_prompt_concepts and instance.valid_true_answer:
                used += 1
                [rows, y_true_matrix, y_pred_matrix] = instance.print(label_ids)
                instance_rows = instance_rows + rows
                if dataset_y_true_matrix is None:
                    dataset_y_true_matrix = y_true_matrix
                    dataset_y_pred_matrix = y_pred_matrix
                else:
                    dataset_y_true_matrix = np.concatenate([dataset_y_true_matrix, y_true_matrix])
                    dataset_y_pred_matrix = np.concatenate([dataset_y_pred_matrix, y_pred_matrix])
            else:
                skipped += 1
                [rows, y_true_matrix, y_pred_matrix] = instance.print(label_ids)
                instance_rows = instance_rows + rows

        assert total == skipped + used
        dataset_result_row = ([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '] +
                              Evaluator.f1(
                                  y_true=dataset_y_true_matrix,
                                  y_pred=dataset_y_pred_matrix, average='weighted',
                                  labels=label_ids)
                              )

        return [header] + [dataset_result_row] + instance_rows


class NelDataset(Dataset):
    def add_instance(self, id_: int, original_prompt: str, true_prompt: str, true_answer: str,
                     predicted_answer: str, actual_predicted_answer: str):
        instance = NelInstance(
            id_=id_,
            original_prompt=original_prompt,
            true_prompt=true_prompt,
            true_answer=true_answer,
            predicted_answer=predicted_answer,
            actual_predicted_answer=actual_predicted_answer)
        return instance


class FsaDataset(Dataset):
    def add_instance(self, id_: int, original_prompt: str, true_prompt: str, true_answer: str,
                     predicted_answer: str, actual_predicted_answer: str):
        instance = FsaInstance(
            id_=id_,
            original_prompt=original_prompt,
            true_prompt=true_prompt,
            true_answer=true_answer,
            predicted_answer=predicted_answer,
            actual_predicted_answer=actual_predicted_answer)
        return instance


class NutritionDataset(Dataset):
    def add_instance(self, id_: int, original_prompt: str, true_prompt: str, true_answer: str,
                     predicted_answer: str, actual_predicted_answer: str):
        instance = NutritionInstance(
            id_=id_,
            original_prompt=original_prompt,
            true_prompt=true_prompt,
            true_answer=true_answer,
            predicted_answer=predicted_answer,
            actual_predicted_answer=actual_predicted_answer)
        return instance
