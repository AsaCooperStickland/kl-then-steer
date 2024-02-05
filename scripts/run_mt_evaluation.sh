!/bin/bash

base_path=/scratch/alc9734/latent-adversarial-training/results ;
# for c in 1000 4000 16000; do
# for m in ${base_path}/run_large_scale_concept_1/checkpoint-${c} ; do sbatch submit_mt.sh $m large_scale_concept_$c ; done
# for m in ${base_path}/run_persuasion_0.5/checkpoint-${c} ; do sbatch submit_mt.sh $m persuasion_0.5_$c ; done;
# for m in ${base_path}/run_no_steer_full/checkpoint-${c} ; do sbatch submit_mt.sh $m no_steer_$c ; done;
# for m in ${base_path}/run_2/checkpoint-${c} ; do sbatch submit_mt.sh $m refusal_$c ; done;
# done

# for m in /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf ; do sbatch submit_mt.sh $m llama-2-chat7b ; done;
# for m in /vast/work/public/ml-datasets/llama-2/Llama-2-13b-chat-hf ; do sbatch submit_mt.sh $m llama-2-chat13b ; done;
# for m in ${base_path}/run2_persuasion_0.5 ; do sbatch submit_mt.sh $m persuasion_0.5 ; done;
for m in ${base_path}/run2_no_steer ; do sbatch submit_mt.sh $m no_steer ; done;