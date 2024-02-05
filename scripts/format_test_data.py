from lat.format_utils import QuestionAugmenter
from lat.utils import jailbreaks_path


if __name__ == "__main__":
    augmenter = QuestionAugmenter(dataset_path="datasets", 
                                  jailbreaks_path=jailbreaks_path,
                                  jinja_directory="/scratch/alc9734/llm-jailbreaks/prompts/wei-jailbreaks/")
    augmenter.augment_questions()