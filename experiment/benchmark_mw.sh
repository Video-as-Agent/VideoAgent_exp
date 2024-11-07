# usage: bash benchmark.sh <GPU_ID> <PATH_TO_BENCHMARK.PY>
for task in "button-press-topdown-v2-goal-observable" "handle-press-v2-goal-observable" "hammer-v2-goal-observable" "basketball-v2-goal-observable" "button-press-v2-goal-observable"
do 
    python benchmark_mw.py --env_name $task --n_exps 25 --ckpt_dir "../ckpts/metaworld" --milestone 3053083  --result_root "../results/3053083_vlm_replan_update_3" 
done
