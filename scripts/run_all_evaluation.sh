!/bin/bash

base_path=/scratch/alc9734/latent-adversarial-training/results ;
# for c in 1000 ; do # 4000 16000; do
# for m in ${base_path}/run_large_scale_concept_1/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# for m in ${base_path}/run_persuasion_0.5/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# for m in ${base_path}/run_no_steer_full/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# for m in ${base_path}/run_2/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# done

for m in /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf ; do for e in refusal emotions_happiness emotions_anger emotions_sadness emotions_fear emotions_disgust emotions_surprise ; do sbatch submit_eval.sh $m $e /scratch/alc9734/latent-adversarial-training/datasets/ ; done; done ;
for m in /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf ; do for e in large_scale_concept_crime large_scale_concept_criminal large_scale_concept_shoplifting large_scale_concept_murder large_scale_concept_fraud large_scale_concept_spearphishing large_scale_concept_burglary large_scale_concept_blackmail large_scale_concept_prison large_scale_concept_evil large_scale_concept_bratty large_scale_concept_sleazy large_scale_concept_nasty large_scale_concept_cold and calculating large_scale_concept_scheming large_scale_concept_disgusting large_scale_concept_amoral large_scale_concept_exploitative large_scale_concept_mean large_scale_concept_know-it-all large_scale_concept_deceitful large_scale_concept_manipulative large_scale_concept_ruthless large_scale_concept_vindictive large_scale_concept_callous large_scale_concept_unscrupulous large_scale_concept_malicious large_scale_concept_greedy large_scale_concept_heartless large_scale_concept_cruel large_scale_concept_vengeful large_scale_concept_selfish large_scale_concept_unethical large_scale_concept_treacherous large_scale_concept_violent large_scale_concept_sadistic ; do sbatch submit_eval.sh $m $e /scratch/alc9734/latent-adversarial-training/datasets/ ; done; done ;
for m in /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf ; do for e in refusal_data_A_B_cropped refusal_data_full_answers refusal_data_A_B_question_pairs filtered_questions_style_question_pairs ; do sbatch submit_eval.sh $m $e /scratch/alc9734/latent-adversarial-training/lat/finetuning/steering_data ; done; done ;


# for m in /vast/work/public/ml-datasets/llama-2/Llama-2-13b-chat-hf ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# for m in ${base_path}/run2_persuasion_0.5 ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# for m in ${base_path}/run2_no_steer ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;