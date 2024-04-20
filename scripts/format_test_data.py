from lat.utils import jailbreaks_path
from lat.format_utils import QuestionAugmenter, RefusalAugmenter, ProbeQuestionAugmenter


if __name__ == "__main__":
    augmenter = QuestionAugmenter(dataset_path="datasets", 
                                  jailbreaks_path=jailbreaks_path,
                                  jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/",
                                  output_dataset_path="datasets/refusal")
    augmenter.augment_questions()
    augmenter = QuestionAugmenter(dataset_path="datasets", 
                                  jailbreaks_path=jailbreaks_path,
                                  jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/",
                    output_dataset_path="datasets/probing/testing")
    # steering_types = "refusal_data_A_B_cropped refusal_data_full_answers refusal_data_A_B_question_pairs filtered_questions_style_question_pairs".split(" ")
    augmenter.augment_questions()
    output_file_path = "question"
    augmenter = ProbeQuestionAugmenter(dataset_path="FastChat/fastchat/llm_judge/data/mt_bench/", 
                    jailbreaks_path=jailbreaks_path,
                    jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/", jinja_subset="test", questions_file=output_file_path,
                    output_dataset_path="datasets/probing/mt_bench", assume_normal=True)
    augmenter.augment_questions()

    # refusal_augmenter = RefusalAugmenter(dataset_path="lat/finetuning/steering_data",
    #                                     jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/")
    # steering_types = "refusal_data_A_B_cropped refusal_data_full_answers".split(" ")
    # for steering_type in steering_types:
    #     steering_input_file = f"{steering_type}.json"
    #     steering_output_file = f"{steering_type}_jinja_augmented.json"
    #     refusal_augmenter.augment_refusal(steering_input_file, steering_output_file)
