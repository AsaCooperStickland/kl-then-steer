from lat.format_utils import QuestionAugmenter, RefusalAugmenter
from lat.utils import jailbreaks_path


if __name__ == "__main__":
    augmenter = QuestionAugmenter(dataset_path="datasets", 
                                  jailbreaks_path=jailbreaks_path,
                                  jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/")
    augmenter.augment_questions()
    refusal_augmenter = RefusalAugmenter(dataset_path="lat/finetuning/steering_data",
                                        jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/")
    
    # steering_types = "refusal_data_A_B_cropped refusal_data_full_answers refusal_data_A_B_question_pairs filtered_questions_style_question_pairs".split(" ")
    steering_types = "refusal_data_A_B_cropped refusal_data_full_answers".split(" ")
    for steering_type in steering_types:
        steering_input_file = f"{steering_type}.json"
        steering_output_file = f"{steering_type}_jinja_augmented.json"
        refusal_augmenter.augment_refusal(steering_input_file, steering_output_file)