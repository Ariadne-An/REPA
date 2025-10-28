#!/bin/bash

echo "开始监控评估进度..."
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================"

while true; do
    clear
    echo "================================"
    echo "REPA训练进度评估监控"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================"
    echo
    
    # 检查生成的图片数量
    echo "📊 图片生成进度:"
    for step in step6k step12k step18k step24k; do
        dir="eval_outputs/training_progress/samples_${step}"
        if [ -d "$dir" ]; then
            count=$(ls "$dir"/*.png 2>/dev/null | wc -l)
            if [ "$count" -eq 300 ]; then
                echo "  ✅ ${step}: ${count}/300"
            else
                echo "  ⏳ ${step}: ${count}/300"
            fi
        else
            echo "  ⏸  ${step}: 未开始"
        fi
    done
    echo
    
    # 检查评估脚本进程
    eval_procs=$(ps aux | grep "evaluate_training_progress.py" | grep -v grep | wc -l)
    if [ "$eval_procs" -gt 0 ]; then
        echo "✅ 评估进程运行中 ($eval_procs processes)"
    else
        echo "❌ 评估进程已停止"
    fi
    echo
    
    # 检查最终结果文件
    if [ -f "eval_outputs/training_progress/evaluation_results.json" ]; then
        echo "🎉 评估完成！"
        echo
        echo "最终结果:"
        cat eval_outputs/training_progress/evaluation_results.json
        echo
        break
    fi
    
    # 显示最新日志
    echo "📝 最新日志 (最后5行):"
    tail -5 eval_outputs/training_progress/evaluation.log 2>/dev/null | grep -v "^$" || echo "  等待日志..."
    echo
    echo "---"
    echo "下次更新: 2分钟后 (按Ctrl+C退出监控)"
    
    sleep 120
done

echo "监控结束"
