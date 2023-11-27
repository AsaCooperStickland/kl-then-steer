import json
import torch
from tqdm import tqdm

from lat.generate_steering_vectors import generate_single_steering_vector


def get_vec(directory, layer, vec_type=""):
    return torch.load(f"{directory}/vec{vec_type}_layer_{layer}.pt")


def generate_with_vector(model, questions, directory, saved_vector, example=None, question_type="", temperature=0.0):
    if "vanilla" in question_type:
        layers = ["n/a"]
        multipliers = ["n/a"]
    else:
        layers = [10, 12, 14, 16]
        # [x / 10 for x in range(-32, 32, 4)]
        multipliers = [-3.2, -1.6, -0.8, -0.4]
    max_new_tokens = 400
    if not saved_vector and "vanilla" not in question_type:
        assert example is not None
        activations = generate_single_steering_vector(model, example)
    else:
        activations = None
    model.set_save_internal_decodings(False)
    all_results = []

    batch_size = 8
    vec_type2name = {"": "activations", "_value": "value",
                     "_query": "query", "n/a": "n/a"}
    if "vanilla" in question_type:
        vec_types = ["n/a"]
    else:
        vec_types = ["", "_value", "_query"]

    for layer in layers:
        layer_results = []
        # for multiplier in tqdm(multipliers):
        for multiplier in multipliers:
            answers = []
            for vec_type in vec_types:
                vec_name = vec_type2name[vec_type]
                model.reset_all()
                if saved_vector:
                    vec = get_vec(directory, layer, vec_type)
                elif "vanilla" in question_type:
                    vec = None
                else:
                    vec = activations[layer]
                if vec is not None:
                    if vec_type == "_value":
                        model.set_add_value_activations(
                            layer, multiplier * vec.cuda())
                    elif vec_type == "_query":
                        model.set_add_query_activations(
                            layer, multiplier * vec.cuda())
                    else:
                        model.set_add_activations(
                            layer, multiplier * vec.cuda())

                # Batch questions
                if "source" in questions[0]:
                    source_exists = True
                else:
                    source_exists = False
                for i in tqdm(range(0, len(questions), batch_size)):
                    # batch, and handle the case where we have less than batch_size questions
                    batched_questions = [
                        q["question"] for q in questions[i: min(i + batch_size, len(questions))]]
                    batched_categories = [
                        q["category"] for q in questions[i: min(i + batch_size, len(questions))]]
                    if source_exists:
                        batched_sources = [
                            q["source"] for q in questions[i: min(i + batch_size, len(questions))]]
                    else:
                        batched_sources = None
                    generated_texts = model.generate_text(
                        batched_questions, max_new_tokens=max_new_tokens, temperature=temperature)

                    for question, category, text in zip(batched_questions, batched_categories, generated_texts):
                        text = text.split("[/INST]")[-1].strip()
                        print(f"Question: {question}")
                        print(f"Category: {category}")
                        print(f"Answer: {text}")
                        print(
                            f"Settings: layer {layer}, multiplier {multiplier}, directory {directory}, question_type {question_type}")
                        if batched_sources:
                            source = batched_sources[batched_questions.index(
                                question)]
                            answers.append({"question": question, "answer": text,
                                            "category": category, "source": source, "vec_name": vec_name})
                        else:
                            answers.append(
                                {"question": question, "answer": text, "category": category, "vec_name": vec_name})
                        layer_name = "" if layer == "n/a" else f"layer_{layer}"
                        multiplier_name = "" if multiplier == "n/a" else f"multiplier_{multiplier}"
                        with open(f"{directory}/{question_type}results{layer_name}{multiplier_name}_temp.json", "w") as jfile:
                            json.dump(answers, jfile)

            layer_results.append(
                {"multiplier": multiplier, "answers": answers})
        all_results.append({"layer": layer, "results": layer_results})

    with open(f"{directory}/{question_type}results.json", "w") as jfile:
        json.dump(all_results, jfile)
