!/bin/bash

base_path=/scratch/alc9734/latent-adversarial-training/results ;
# for c in 1000 ; do # 4000 16000; do
# for m in ${base_path}/run_large_scale_concept_1/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# for m in ${base_path}/run_persuasion_0.5/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# for m in ${base_path}/run_no_steer_full/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# for m in ${base_path}/run_2/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# done

# for m in /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
for m in /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf ; do for e in large_scale_concept_crime large_scale_concept_criminal large_scale_concept_shoplifting large_scale_concept_murder large_scale_concept_fraud large_scale_concept_spearphishing large_scale_concept_burglary large_scale_concept_blackmail large_scale_concept_prison ; do sbatch submit_eval.sh $m $e /scratch/alc9734/latent-adversarial-training/datasets/ ; done; done ;
# for m in /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf ; do for e in large_scale_concept_crime large_scale_concept_criminal large_scale_concept_shoplifting large_scale_concept_murder large_scale_concept_fraud large_scale_concept_spearphishing large_scale_concept_burglary large_scale_concept_blackmail large_scale_concept_prison ; do sbatch submit_eval.sh $m $e /scratch/alc9734/latent-adversarial-training/lat/finetuning/steering_data ; done; done ;


# for m in /vast/work/public/ml-datasets/llama-2/Llama-2-13b-chat-hf ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# for m in ${base_path}/run2_persuasion_0.5 ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# for m in ${base_path}/run2_no_steer ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;