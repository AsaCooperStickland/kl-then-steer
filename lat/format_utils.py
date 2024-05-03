import json
import jsonlines
import random
import glob
import os
from collections import defaultdict

from lat.utils import system_prompt

from llmtuner.data.template import templates as llmtuner_templates
from llmtuner.data.utils import Role
from transformers import AutoTokenizer

random.seed(27)

llama3_tokenizer = AutoTokenizer.from_pretrained("/scratch/al6759/lat/Meta-Llama-3-8B-Instruct")
# llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


def json_to_questions(json_path: str) -> list:
    questions = []
    with open(json_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            questions.append(data['question'])
    return questions


def prompt_format(instruction, template="llama2chatsimple", alternative_system_prompt=None):
    assert template in ["llama2chatsimple", "chatml", "llama3"]
    system_prompt_used = system_prompt if alternative_system_prompt is None else alternative_system_prompt
    if template == "llama2chatsimple":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        dialog_content = B_SYS + system_prompt_used + E_SYS + instruction.strip()
        dialog_content = f"{B_INST} {dialog_content.strip()} {E_INST}"
    elif template == "chatml":
        B_INST, E_INST = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"
        B_SYS, E_SYS = "<|im_start|>system\n", "<|im_end|>\n"
        dialog_content = B_SYS + system_prompt_used + E_SYS + instruction.strip()
        dialog_content = f"{B_INST} {dialog_content.strip()} {E_INST}"
    elif template == "llama3":
        messages = [
            {'role': Role.USER.value, 'content': instruction},
            {'role': Role.ASSISTANT.value, 'content': ''},
        ]
        encoded = llmtuner_templates['llama3'].encode_oneturn(llama3_tokenizer, messages)
        dialog_content = llama3_tokenizer.decode(encoded[0])
    else:
        raise ValueError
        
    return dialog_content


def check_source_for_refusal(text):
    if ("gpt" in text or "claude" in text or "mistral" in text or "Mixtral" in text):
        return True
    return False


class QuestionReader:
    def __init__(self, base_path):
        self.base_path = base_path

    def read_initial_questions(self, file_name):
        path = os.path.join(self.base_path, file_name)
        output = []
        with jsonlines.open(path, mode='r') as reader:
            for item in reader:
                if "source" not in item and "question" in item:
                    item["source"] = item["category"]
                elif "source" not in item and "question" not in item:
                    item["source"] = "mt_bench"
                    item["category"] = "mt_bench"
                    item["question"] = item["turns"][0]
                output.append(item)
        return output

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
    def format_questions(self, templates, questions, category, model, extra_diverse=False, augment_all=True):
        formatted_questions = []
        mappings = {}
        random.shuffle(questions)
        question_interval = len(questions) // len(templates)
        for i, template_info in enumerate(templates):
            start_index = i * question_interval
            end_index = start_index + question_interval
            if augment_all:
                start_index = 0
                end_index = len(questions)
            template_name, template_prompt = template_info
            category_key = f"{template_name}_{category}_{model}"
            category_key += "_extra_diverse" if extra_diverse else ""
            for item in questions[start_index:end_index]:
                augmented_question = self.format_example(template_prompt, item["question"])
                mappings[augmented_question] = item["question"]
                new_item = {"question": augmented_question, "category": category_key, "source": category_key}
                for k, v in item.items():
                    if k not in new_item:
                        new_item[k] = v
                formatted_questions.append(new_item)
        return formatted_questions, mappings

    def get_category_keys(self, templates, category, model, extra_diverse=False):
        category_keys = []
        for template_info in templates:
            template_name, _ = template_info
            category_key = f"{template_name}_{category}_{model}"
            category_key += "_extra_diverse" if extra_diverse else ""
            category_keys.append(category_key)
        return category_keys

    def format_example(self, template, question):
        return template.replace("{prompt}", question)


class AdditionFormatter(Formatter):
    def format_example(self, template, question):
        return template + question


class QuestionAugmenter:
    def __init__(self, dataset_path, jailbreaks_path, jinja_directory, jinja_subset="test", output_dataset_path=None):
        self.reader = QuestionReader(dataset_path)
        self.dataset_path = dataset_path
        self.output_dataset_path = output_dataset_path if output_dataset_path else dataset_path
        if not os.path.exists(self.output_dataset_path):
            os.makedirs(self.output_dataset_path, exist_ok=True)
        self.jinja_files = FileLoader.load_jinja_files(jinja_directory)
        random.shuffle(self.jinja_files)
        if jinja_subset == "test":
            self.jinja_files = self.jinja_files[:5]
        else:
            self.jinja_files = self.jinja_files[5:]
        with open(jailbreaks_path, "r") as f:
            self.jailbreaks = json.load(f)
        self.jailbreaks = [(item["name"], item["prompt"]) for item in self.jailbreaks]
        self.jinja_formatter = Formatter()
        self.addition_formatter = AdditionFormatter()
        self.mapping = {}
        self.initial_questions = self.reader.read_initial_questions("refusal/filtered_questions.jsonl")
        misc_questions_file = "refusal/misc.txt"
        with open(os.path.join(dataset_path, misc_questions_file), "r") as file:
            misc_questions = [line.strip() for line in file]
            self.initial_questions.extend([{"question": item, "source": "misc", "category": "misc"} for item in misc_questions])
        self.augmented_categories = ["illegal_activity", "race_bias", "nationality_bias", "misc"]
    
    def get_all_category_keys(self):
        category_key_store = defaultdict(list)
        vanilla_categories = list(set([item["category"] for item in self.initial_questions]))
        category_key_store["vanilla_categories"] = vanilla_categories
        for category in self.augmented_categories:
            vanilla_jinja_keys = self.jinja_formatter.get_category_keys(self.jinja_files, category, "initial")
            category_key_store["vanilla_jinja_categories"].extend(vanilla_jinja_keys)
            category_key_store["all_jinja_categories"].extend(vanilla_jinja_keys)
            vanilla_jailbreak_keys = self.addition_formatter.get_category_keys(self.jailbreaks, category, "initial")
            category_key_store["vanilla_jailbreak_categories"].extend(vanilla_jailbreak_keys)
            category_key_store["all_jailbreak_categories"].extend(vanilla_jailbreak_keys)
        all_rephrased_category_keys = []
        model_rephrased_category_keys = defaultdict(list)
        for category in self.augmented_categories:
            for model in ["gpt-4", "gpt-3.5-turbo-16k-0613"]:
                for extra_diverse in [True, False]:
                    new_category_key = f"{category}_{model}" if not extra_diverse else f"{category}_{model}_extra_diverse"
                    all_rephrased_category_keys.append(new_category_key)
                    model_rephrased_category_keys[model].append(new_category_key)
                    jinja_category_keys = self.jinja_formatter.get_category_keys(self.jinja_files, category, model, extra_diverse=extra_diverse)
                    category_type = f"{model}_jinja" if not extra_diverse else f"{model}_extra_diverse_jinja"
                    category_key_store[category_type].extend(jinja_category_keys)
                    category_key_store["all_jinja_categories"].extend(jinja_category_keys)
                    jailbreak_category_keys = self.addition_formatter.get_category_keys(self.jailbreaks, category, model, extra_diverse=extra_diverse)
                    category_type = f"{model}_jailbreak" if not extra_diverse else f"{model}_extra_diverse_jailbreak"
                    category_key_store[category_type].extend(jailbreak_category_keys)
                    category_key_store["all_jailbreak_categories"].extend(jailbreak_category_keys)
        category_key_store["all_rephrased_categories"] = all_rephrased_category_keys
        for model, category_keys in model_rephrased_category_keys.items():
            category_key_store[f"{model}_rephrased_categories"] = category_keys
        return category_key_store


    def augment_questions(self):
        print(self.jinja_files)
        ultra_filtered_questions = [item for item in self.initial_questions if item["category"] in self.augmented_categories]
        print(f"Ultra filtered questions: {len(ultra_filtered_questions)}")
        ultra_filtered_questions_file_path = os.path.join(self.dataset_path, "refusal/ultra_filtered_questions.jsonl")
        with jsonlines.open(ultra_filtered_questions_file_path, mode='w') as writer:
            writer.write_all(ultra_filtered_questions)
        
        for category in self.augmented_categories:
            questions = [item for item in self.initial_questions if item["category"] == category]
            augmented_questions, mapping = self.jinja_formatter.format_questions(self.jinja_files, questions, category, "initial")
            self.mapping.update(mapping)
            self.initial_questions.extend(augmented_questions)
            jailbreak_questions, mapping = self.addition_formatter.format_questions(self.jailbreaks, questions, category, "initial")
            self.mapping.update(mapping)
            self.initial_questions.extend(jailbreak_questions)

        for category in self.augmented_categories:
            for model in ["gpt-4", "gpt-3.5-turbo-16k-0613", "mistralai/Mixtral-8x7B-Instruct-v0.1"]:
                for extra_diverse in [True, False]:
                    questions = self.reader.read_rephrased_questions(model, category, extra_diverse=extra_diverse, test_set=True)
                    print(f"questions and category {category} for model {model} using extra_diverse {extra_diverse}: {len(questions)}")
                    new_category_key = f"{category}_{model}" if not extra_diverse else f"{category}_{model}_extra_diverse"
                    questions = [{"question": item["question"], "source": new_category_key, "category": new_category_key} for item in questions]
                    self.initial_questions.extend(questions)
                    jinja_questions, mapping = self.jinja_formatter.format_questions(self.jinja_files, questions, category, 
                                                                                     model, extra_diverse=extra_diverse, augment_all=False)
                    self.mapping.update(mapping)
                    self.initial_questions.extend(jinja_questions)
                    jailbreak_questions, mapping = self.addition_formatter.format_questions(self.jailbreaks, questions, category, 
                                                                                            model, extra_diverse=extra_diverse, augment_all=False)
                    self.mapping.update(mapping)
                    self.initial_questions.extend(jailbreak_questions)

        # Save the augmented questions
        print(f"initial_questions after augmentation: {len(self.initial_questions)}")
        augmented_questions_file_path = os.path.join(self.output_dataset_path, "augmented_questions.jsonl")
        with jsonlines.open(augmented_questions_file_path, mode='w') as writer:
            writer.write_all(self.initial_questions)
        self.save_mappings()

    def save_mappings(self):
        with open(os.path.join(self.dataset_path, "refusal/question_mappings.json"), "w") as file:
            json.dump(self.mapping, file, indent=4)


class ProbeQuestionAugmenter:
    def __init__(self, dataset_path, jailbreaks_path, jinja_directory, jinja_subset="test", questions_file="probing/training/training_persuasion0.50_refusal0.50", output_dataset_path=None, assume_toxic=False, assume_normal=False, augment_subset=False):
        self.reader = QuestionReader(dataset_path)
        self.dataset_path = dataset_path
        self.output_dataset_path = output_dataset_path if output_dataset_path else dataset_path
        if not os.path.exists(self.output_dataset_path):
            os.makedirs(self.output_dataset_path, exist_ok=True)
        self.jinja_files = FileLoader.load_jinja_files(jinja_directory)
        random.shuffle(self.jinja_files)
        if jinja_subset == "test":
            self.jinja_files = self.jinja_files[:5]
        else:
            self.jinja_files = self.jinja_files[5:]
        with open(jailbreaks_path, "r") as f:
            self.jailbreaks = json.load(f)
        self.jailbreaks = [(item["name"], item["prompt"]) for item in self.jailbreaks]
        self.jinja_formatter = Formatter()
        self.addition_formatter = AdditionFormatter()
        self.mapping = {}
        self.initial_questions = self.reader.read_initial_questions(f"{questions_file}.jsonl")
        if augment_subset:
            misc_questions_file = "refusal/misc.txt"
            with open(os.path.join(dataset_path, misc_questions_file), "r") as file:
                misc_questions = [line.strip() for line in file]
                self.initial_questions.extend([{"question": item, "category": "misc"} for item in misc_questions])
            self.augmented_categories = ["illegal_activity", "race_bias", "nationality_bias", "misc"]
        else:
            self.augmented_categories = set([item["source"] for item in self.initial_questions])
        self.questions_file = questions_file
        self.augment_subset = augment_subset
        self.assume_toxic = assume_toxic
        self.assume_normal = assume_normal
        assert not (self.assume_toxic and self.assume_normal)
    
    def get_all_category_keys(self):
        category_key_store = defaultdict(list)
        if self.assume_toxic:
            toxic_categories = list(set([item["source"] for item in self.initial_questions]))
            normal_categories = []
        elif self.assume_normal:
            toxic_categories = []
            normal_categories = list(set([item["source"] for item in self.initial_questions]))
        else:
            toxic_categories = list(set([item["source"] for item in self.initial_questions if check_source_for_refusal(item["source"])]))
            normal_categories = list(set([item["source"] for item in self.initial_questions if not check_source_for_refusal(item["source"])]))
        category_key_store["toxic_categories"] = toxic_categories
        category_key_store["normal_categories"] = normal_categories
        category_key_store["all_toxic_categories"] = toxic_categories
        category_key_store["all_normal_categories"] = normal_categories
        for category in self.augmented_categories:
            vanilla_jinja_keys = self.jinja_formatter.get_category_keys(self.jinja_files, category, "initial")
            if category in toxic_categories:
                category_key_store["toxic_jinja_categories"].extend(vanilla_jinja_keys)
                category_key_store["all_toxic_categories"].extend(vanilla_jinja_keys)
            else:
                category_key_store["normal_jinja_categories"].extend(vanilla_jinja_keys)
                category_key_store["all_normal_categories"].extend(vanilla_jinja_keys)
            vanilla_jailbreak_keys = self.addition_formatter.get_category_keys(self.jailbreaks, category, "initial")
            if category in toxic_categories:
                category_key_store["toxic_jailbreak_categories"].extend(vanilla_jailbreak_keys)
                category_key_store["all_toxic_categories"].extend(vanilla_jailbreak_keys)
            else:
                category_key_store["normal_jailbreak_categories"].extend(vanilla_jailbreak_keys)
                category_key_store["all_normal_categories"].extend(vanilla_jailbreak_keys)
        return category_key_store


    def augment_questions(self):
        # print(self.jinja_files)
        ultra_filtered_questions = [item for item in self.initial_questions if item["category"] in self.augmented_categories]
        print(f"Ultra filtered questions: {len(ultra_filtered_questions)}")
        ultra_filtered_questions_file_path = os.path.join(self.dataset_path, f"{self.questions_file}_augmented.jsonl")
        with jsonlines.open(ultra_filtered_questions_file_path, mode='w') as writer:
            writer.write_all(ultra_filtered_questions)
        
        for category in self.augmented_categories:
            questions = [item for item in self.initial_questions if item["category"] == category]
            augmented_questions, mapping = self.jinja_formatter.format_questions(self.jinja_files, questions, category, "initial", augment_all=False)
            self.mapping.update(mapping)
            self.initial_questions.extend(augmented_questions)
            jailbreak_questions, mapping = self.addition_formatter.format_questions(self.jailbreaks, questions, category, "initial", augment_all=False)
            self.mapping.update(mapping)
            self.initial_questions.extend(jailbreak_questions)

        # Save the augmented questions
        print(f"initial_questions after augmentation: {len(self.initial_questions)}")
        
        category_keys = self.get_all_category_keys()
        data_dictionary = {}
        for key in category_keys["all_toxic_categories"] + category_keys["all_normal_categories"]:
            data_dictionary[key] = [item for item in self.initial_questions if item["category"] == key]
        augmented_questions_file_path = os.path.join(self.output_dataset_path, f"{self.questions_file}_questions.json")
        print(augmented_questions_file_path)
        with open(augmented_questions_file_path, "w") as file:
            json.dump(data_dictionary, file, indent=4)

    def save_mappings(self):
        with open(os.path.join(self.output_dataset_path, "probing/question_mappings.json"), "w") as file:
            json.dump(self.mapping, file, indent=4)


class RefusalAugmenter:
    def __init__(self, dataset_path, jinja_directory, jinja_subset="train"):
        self.dataset_path = dataset_path
        self.jinja_files = FileLoader.load_jinja_files(jinja_directory)
        random.shuffle(self.jinja_files)
        if jinja_subset == "test":
            self.jinja_files = self.jinja_files[:5]
        else:
            self.jinja_files = self.jinja_files[5:]
        self.jinja_formatter = Formatter()

    def augment_refusal(self, input_filename, output_filename):
        with open(os.path.join(self.dataset_path, input_filename), "r") as f:
            refusal_data = json.load(f)
        
        new_refusal_data, mapping = self.jinja_formatter.format_questions(self.jinja_files, refusal_data, category="n/a", 
                                                                 model="n/a", extra_diverse=False, augment_all=False)
        new_refusal_data_augmented = []
        for item in new_refusal_data:
            # print(item["question"], "question")
            old_question = mapping[item["question"]]
            # print(old_question, "old question")
            augmented_question = item["question"]
            item["augmented_question"] = augmented_question
            item["question"] = old_question
            new_refusal_data_augmented.append(item)
        
        with open(os.path.join(self.dataset_path, output_filename), "w") as f:
            json.dump(new_refusal_data, f)
        
        
