#!/bin/bash
task_list="evaluate_tasks.txt"
> "$task_list"  # Clear the task list

# Get all directories which have a run output (output.json) but no evaluation results (metrics.json)
run_paths=$(find output -type f -name "output.json" | while read -r file; do
    dir=$(dirname "$file")
    if [[ ! -f "$dir/metrics.json" ]]; then
        echo "$dir"
    fi
done)

for run_path in "${run_paths[@]}"; do
    echo "$run_path" >> "$task_list"
done

# Count the number of tasks
n_tasks=$(wc -l < "$task_list")

# Print command to submit the job array, where each job processes a different run
echo "Tasks: $n_tasks"
echo "Task file: $task_list"
echo "Use the following command to start the job array. See marcel_evaluation/runner.py for a list of optional arguments."
echo sbatch --array=0-$(($n_tasks - 1))%20 scripts/evaluate.sh --model neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8 --metrics GeneratedAnswerLength,ReferenceAnswerLength,ContextLength,MeanReciprocalRank,PrecisionAtCutoff,RecallAtCutoff,ContextPrecision,ContextRecall,CorpusBLEU,ROUGE,BERTScore,AnswerSimilarity,AnswerFaithfulness,AnswerRelevance,NonAnswerCritic
