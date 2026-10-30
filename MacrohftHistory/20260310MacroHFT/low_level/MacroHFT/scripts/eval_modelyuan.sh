#!/bin/bash
# Simple batch evaluation for all best_modelyuan models

echo "Starting batch evaluation..."

# Find all best_model.pkl files under best_modelyuan
find result/low_level/ETHUSDT/best_modelyuan -name "best_model.pkl" | while read model_path; do
    echo ""
    echo "===================================================================="
    echo "Evaluating: $model_path"
    echo "===================================================================="
    
    # Extract label from path (e.g., slope/1 -> label_1)
    if [[ $model_path =~ slope/([0-9]+) ]]; then
        label="label_${BASH_REMATCH[1]}"
        clf="slope"
    elif [[ $model_path =~ vol/([0-9]+) ]]; then
        label="label_${BASH_REMATCH[1]}"
        clf="vol"
    else
        echo "Could not determine label from path, skipping..."
        continue
    fi
    
    echo "Classifier: $clf, Label: $label"
    
    # Evaluate on validation set
    echo ">> Validation set..."
    python -m RL.agent.eval_low_level \
        --pretrained_model "$model_path" \
        --dataset "ETHUSDT" \
        --clf "$clf" \
        --label "$label" \
        --num_actions 2 \
        --eval_set "val" \
        --device "cuda:0"
    
    # Evaluate on test set
    echo ">> Test set..."
    python -m RL.agent.eval_low_level \
        --pretrained_model "$model_path" \
        --dataset "ETHUSDT" \
        --clf "$clf" \
        --label "$label" \
        --num_actions 2 \
        --eval_set "test" \
        --device "cuda:0"
    
    echo "Completed: $model_path"
done

echo ""
echo "===================================================================="
echo "All evaluations completed!"
echo "Generating summary..."
echo "===================================================================="

python scripts/summarize_eval_results.py

echo ""
echo "Done! Check logs/low_level/ETHUSDT/ for detailed logs"
echo "Results summary: result/low_level/ETHUSDT/evaluation_summary.csv"
