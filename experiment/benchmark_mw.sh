# usage: bash benchmark.sh <GPU_ID> <PATH_TO_BENCHMARK.PY>
for task in "button-press-topdown-v2-goal-observable" "handle-press-v2-goal-observable" "hammer-v2-goal-observable" "basketball-v2-goal-observable" "button-press-v2-goal-observable"
do 
    CUDA_VISIBLE_DEVICES=$1 accelerate launch benchmark_mw.py --env_name $task --n_exps 25 --ckpt_dir "../ckpts/metaworld" --milestone 305  --result_root "../results/305_VideoAgent_mw" 
done

python org_results_mw.py --results_root "../results/results_VideoAgent_mw" 
