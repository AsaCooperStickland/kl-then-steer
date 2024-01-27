!/bin/bash

base_path=/scratch/alc9734/latent-adversarial-training/results ;
for c in 1000 4000 ; do
for m in ${base_path}/run_large_scale_concept_1/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e jail ; done; done ;
for m in ${base_path}/run_persuasion0.5/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e jail ; done; done ;
for m in ${base_path}/run_no_steer_full/checkpoint-${c} ; do for e in refusal emotions_happiness emotions_anger ; do sbatch submit_eval.sh $m $e jail ; done; done ;
done