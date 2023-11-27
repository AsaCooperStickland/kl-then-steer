import json
import fire


def main(directory, layer, multiplier, answer_id, results_name=""):
    
    layers = [10, 12, 14, 16]
    layer_to_index = dict([(l, i) for i, l in enumerate(layers)])
    multipliers = [-3.2, -1.6, -0.8, -0.4] # [x / 10 for x in range(-32, 32, 4)]
    multiplier_to_index = dict([(m, i) for i, m in enumerate(multipliers)])
    with open(f"{directory}/{results_name}results.json", "r") as jfile:
        all_results = json.load(jfile)
    if all_results[0]["layer"] == "n/a":
        print(all_results[0]["results"][0]["answers"][answer_id])
        return
    layer_results = all_results[layer_to_index[layer]]
    multiplier_results = layer_results["results"][multiplier_to_index[multiplier]]
    answer = multiplier_results["answers"][answer_id]
    print(answer)

if __name__ == "__main__":
    fire.Fire(main)