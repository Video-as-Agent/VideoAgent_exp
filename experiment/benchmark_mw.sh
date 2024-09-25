# usage: bash benchmark.sh <GPU_ID> <PATH_TO_BENCHMARK.PY>
for task in "button-press-topdown-v2-goal-observable" "handle-press-v2-goal-observable" "hammer-v2-goal-observable" "basketball-v2-goal-observable" "button-press-v2-goal-observable"
do 
    python benchmark_mw.py --env_name $task --n_exps 25 --ckpt_dir "../ckpts/metaworld" --milestone 3053083  --result_root "../results/3053083_vlm_replan_update_3" 
done

# for task in "door-open-v2-goal-observable" "door-close-v2-goal-observable" "basketball-v2-goal-observable" "shelf-place-v2-goal-observable" "button-press-v2-goal-observable" "button-press-topdown-v2-goal-observable" "faucet-close-v2-goal-observable" "faucet-open-v2-goal-observable" "handle-press-v2-goal-observable" "hammer-v2-goal-observable" "assembly-v2-goal-observable"
# do 
#     accelerate launch --num_processes=1 benchmark_mw.py --env_name $task --n_exps 25 --ckpt_dir "../ckpts/metaworld" --milestone 3051242 --result_root "../results/iteration_2_305_124_frame_cond_of" 
# done

# for task in "door-open-v2-goal-observable" "door-close-v2-goal-observable" "basketball-v2-goal-observable" "shelf-place-v2-goal-observable" "button-press-v2-goal-observable" "button-press-topdown-v2-goal-observable" "faucet-close-v2-goal-observable" "faucet-open-v2-goal-observable" "handle-press-v2-goal-observable" "hammer-v2-goal-observable" "assembly-v2-goal-observable"
# do 
#     accelerate launch --num_processes=1 benchmark_mw.py --env_name $task --n_exps 25 --ckpt_dir "../ckpts/metaworld" --milestone 30566 --result_root "../results/iteration_2_305_66_of" 
# done

# for task in "door-open-v2-goal-observable" "door-close-v2-goal-observable" "basketball-v2-goal-observable" "shelf-place-v2-goal-observable" "button-press-v2-goal-observable" "button-press-topdown-v2-goal-observable" "faucet-close-v2-goal-observable" "faucet-open-v2-goal-observable" "handle-press-v2-goal-observable" "hammer-v2-goal-observable" "assembly-v2-goal-observable"
# do 
#     accelerate launch --num_processes=1 benchmark_mw.py --env_name $task --n_exps 25 --ckpt_dir "../ckpts/metaworld" --milestone 30573 --result_root "../results/iteration_2_305_73_of" 
# done

#CUDA_VISIBLE_DEVICES=1 python benchmark_mw_ff.py --env_name "assembly-v2-goal-observable" --n_exps 25 --ckpt_dir "../ckpts/metaworld" --milestone 276 --result_root "../results/results_AVDC_mw_flowfeedback"
# "door-open-v2-goal-observable" "door-close-v2-goal-observable" "basketball-v2-goal-observable" "shelf-place-v2-goal-observable" "button-press-v2-goal-observable"