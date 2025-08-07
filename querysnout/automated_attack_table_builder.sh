#!/bin/bash

if [ $1 = "insurance" ]
then
        SIZE=1000
else
        SIZE=8000
fi

echo $SIZE

for R in 0 1 2 3 4
do
	for SEARCH_TYPE in evolutionary random
	do
		for DATASET_SAMPLER in without_replacement exact
		do
			python main.py  --dataset_path=datasets/$1 \
				--dataset_name=$1 \
				--num_target_users=100 \
				--num_test_samples=500 \
				--num_datasets=3000 \
				--eval_fraction=0.33333 \
				--dataset_sampler=$DATASET_SAMPLER \
				--target_dataset_size=$SIZE \
				--num_procs=$2  \
				--population_size=100 \
				--num_queries=100  \
				--search_type=$SEARCH_TYPE \
			     	--qbs_type=table-builder \
				--num_attributes=5 \
				--p_copy=0.025 \
				--p_modify=0.025 \
				--num_generations=200  \
				--repetition=$R \
				--verbose=False
		done
	done
done
