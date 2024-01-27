import json
import jsonlines
import random
import glob
import os

from lat.utils import jailbreaks_path
random.seed(27)


class QuestionReader:
    def __init__(self, base_path):
        self.base_path = base_path

    def read_initial_questions(self, file_name):
        path = os.path.join(self.base_path, file_name)
        with jsonlines.open(path, mode='r') as reader:
            return [item for item in reader]

    def read_rephrased_questions(self, model, category, extra_diverse=False, test_set=False):
        extra = "_test" if test_set else ""
        extra += "_extra_diverse" if extra_diverse else ""
        path = os.path.join(self.base_path, f"refusal/{model}_{category}_rephrased_questions{extra}.jsonl")
        if not os.path.exists(path):
            return []
        with jsonlines.open(path, mode='r') as reader:
            return [item for item in reader]


class FileLoader:
    @staticmethod
    def load_jinja_files(directory):
        file_paths = glob.glob(f"{directory}/*.jinja")
        file_contents = []
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                content = file.read()
                file_contents.append((os.path.basename(file_path), content))
        return file_contents


class Formatter:
    def format_questions(self, templates, questions, category, model, extra_diverse=False):
        formatted_questions = []
        mappings = {}
        random.shuffle(questions)
        question_interval = len(questions) // len(templates)
        for i, template_info in enumerate(templates):
            start_index = i * question_interval
            end_index = start_index + question_interval
            template_name, template_prompt = template_info
            category_key = f"{template_name}_{category}_{model}"
            category_key += "_extra_diverse" if extra_diverse else ""
            for item in questions[start_index:end_index]:
                augmented_question = self.format_example(template_prompt, item["question"])
                formatted_questions.append({"question": augmented_question, "category": category_key})
                mappings[augmented_question] = item["question"]
        return formatted_questions, mappings


    def format_example(self, template, question):
        return template.replace("{prompt}", question)


class AdditionFormatter(Formatter):
    def format_example(self, template, question):
        return template + question


class QuestionAugmenter:
    def __init__(self, dataset_path, jailbreaks_path, jinja_directory):
        self.reader = QuestionReader(dataset_path)
        self.dataset_path = dataset_path
        self.jinja_files = FileLoader.load_jinja_files(jinja_directory)
        random.shuffle(self.jinja_files)
        self.jinja_files = self.jinja_files[:5]
        print(self.jinja_files)
        with open(jailbreaks_path, "r") as f:
            self.jailbreaks = json.load(f)
        self.jailbreaks = [(item["name"], item["prompt"]) for item in self.jailbreaks]
        self.jinja_formatter = Formatter()
        self.addition_formatter = AdditionFormatter()
        self.mapping = {}

    def augment_questions(self):
        initial_questions = self.reader.read_initial_questions("refusal/filtered_questions.jsonl")
        augmented_categories = ["illegal_activity", "race_bias", "nationality_bias"]
        for category in augmented_categories:
            questions = [item for item in initial_questions if item["category"] == category]
            augmented_questions, mapping = self.jinja_formatter.format_questions(self.jinja_files, questions, category, "initial")
            self.mapping.update(mapping)
            initial_questions.extend(augmented_questions)
            jailbreak_questions, mapping = self.addition_formatter.format_questions(self.jailbreaks, questions, category, "initial")
            self.mapping.update(mapping)
            initial_questions.extend(jailbreak_questions)

        for category in augmented_categories:
            for model in ["gpt-4", "gpt-3.5-turbo-16k-0613", "mistralai/Mixtral-8x7B-Instruct-v0.1"]:
                for extra_diverse in [True, False]:
                    questions = self.reader.read_rephrased_questions(model, category, extra_diverse=extra_diverse, test_set=True)
                    print(f"questions and category {category} for model {model} using extra_diverse {extra_diverse}: {len(questions)}")
                    new_category_key = f"{category}_{model}" if not extra_diverse else f"{category}_{model}_extra_diverse"
                    questions = [{"question": item["question"], "category": new_category_key} for item in questions]
                    initial_questions.extend(questions)
                    jinja_questions, mapping = self.jinja_formatter.format_questions(self.jinja_files, questions, category, model, extra_diverse=extra_diverse)
                    self.mapping.update(mapping)
                    initial_questions.extend(jinja_questions)
                    jailbreak_questions, mapping = self.addition_formatter.format_questions(self.jailbreaks, questions, category, model, extra_diverse=extra_diverse)
                    self.mapping.update(mapping)
                    initial_questions.extend(jailbreak_questions)

        # Save the augmented questions
        augmented_questions_file_path = os.path.join(self.dataset_path, "refusal/augmented_questions.jsonl")
        with jsonlines.open(augmented_questions_file_path, mode='w') as writer:
            writer.write_all(initial_questions)
        self.save_mappings()

    def save_mappings(self):
        with open(os.path.join(self.dataset_path, "refusal/question_mappings.json"), "w") as file:
            json.dump(self.mapping, file, indent=4)


if __name__ == "__main__":
    augmenter = QuestionAugmenter(dataset_path="datasets", 
                                  jailbreaks_path=jailbreaks_path,
                                  jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/")
    augmenter.augment_questions()