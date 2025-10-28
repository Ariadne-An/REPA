#!/bin/bash

# Monitor evaluation progress

echo "============================================"
echo "REPA Training Progress Evaluation Monitor"
echo "============================================"
echo

while true; do
    clear
    echo "============================================"
    echo "REPA Training Progress Evaluation Monitor"
    echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"
    echo

    # Check generated samples
    echo "Generated Samples:"
    for step in step6k step12k step18k step24k; do
        dir="/workspace/REPA/eval_outputs/training_progress/samples_${step}"
        if [ -d "$dir" ]; then
            count=$(ls -1 "$dir"/*.png 2>/dev/null | wc -l)
            echo "  ${step}: ${count}/300 images"
        else
            echo "  ${step}: Not started"
        fi
    done
    echo

    # Check if results file exists
    if [ -f "/workspace/REPA/eval_outputs/training_progress/evaluation_results.json" ]; then
        echo "✅ Evaluation COMPLETED!"
        echo
        echo "Results:"
        cat /workspace/REPA/eval_outputs/training_progress/evaluation_results.json | python3 -m json.tool 2>/dev/null || cat /workspace/REPA/eval_outputs/training_progress/evaluation_results.json
        break
    fi

    # Show last 10 lines of log
    echo "Recent log (last 10 lines):"
    tail -10 /workspace/REPA/eval_outputs/training_progress/evaluation.log 2>/dev/null | grep -v "^$" || echo "  No log yet"
    echo
    echo "Press Ctrl+C to exit monitoring"
    echo "Next update in 30 seconds..."

    sleep 30
done
