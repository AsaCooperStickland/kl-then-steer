!/bin/bash

base_path=/scratch/alc9734/latent-adversarial-training/results ;
for c in 4000 ; do
# for m in ${base_path}/run_large_scale_concept_1/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
for m in ${base_path}/run_persuasion_0.5/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
# for m in ${base_path}/run_no_steer_full/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;
done

for m in /vast/work/public/ml-datasets/llama-2/Llama-2-7b-chat-hf ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e 1 ; done; done ;