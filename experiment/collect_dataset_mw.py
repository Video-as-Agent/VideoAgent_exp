from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
import metaworld.policies as policies
from utils import collect_video, sample_n_frames
from math import ceil
from tqdm import tqdm
import numpy as np
import imageio
import os
import pickle
import multiprocessing as mp

collection_config = {
    "demos_per_camera": 25,
    "output_path": "../metaworld_dataset_1112_test/",
    "camera_names": ['corner', 'corner2', 'corner3'], ### possible values: "corner3, corner, corner2, topview, behindGripper", None(for random)
    "resolution": (320, 240),
    "discard_ratio": 0.3, ### discard the last {ratio} of the collected videos (preventing failed episodes)
}


included_tasks = ["door-open"]# ["drawer-open", "door-close", "basketball", "shelf-place", "button-press", "button-press-topdown", "faucet-close", "faucet-open", "handle-press", "hammer", "assembly"]
included_tasks = [t + "-v2-goal-observable" for t in included_tasks]
#included_tasks = ['assembly-v2-goal-observable', 'basketball-v2-goal-observable', 'bin-picking-v2-goal-observable', 'box-close-v2-goal-observable', 'button-press-topdown-v2-goal-observable', 'button-press-topdown-wall-v2-goal-observable', 'button-press-v2-goal-observable', 'button-press-wall-v2-goal-observable', 'coffee-button-v2-goal-observable', 'coffee-pull-v2-goal-observable', 'coffee-push-v2-goal-observable', 'dial-turn-v2-goal-observable', 'disassemble-v2-goal-observable', 'door-close-v2-goal-observable', 'door-lock-v2-goal-observable', 'door-open-v2-goal-observable', 'door-unlock-v2-goal-observable', 'hand-insert-v2-goal-observable', 'drawer-close-v2-goal-observable', 'drawer-open-v2-goal-observable', 'faucet-open-v2-goal-observable', 'faucet-close-v2-goal-observable', 'hammer-v2-goal-observable', 'handle-press-side-v2-goal-observable', 'handle-press-v2-goal-observable', 'handle-pull-side-v2-goal-observable', 'handle-pull-v2-goal-observable', 'lever-pull-v2-goal-observable', 'pick-place-wall-v2-goal-observable', 'pick-out-of-hole-v2-goal-observable', 'reach-v2-goal-observable', 'push-back-v2-goal-observable', 'push-v2-goal-observable', 'pick-place-v2-goal-observable', 'plate-slide-v2-goal-observable', 'plate-slide-side-v2-goal-observable', 'plate-slide-back-v2-goal-observable', 'plate-slide-back-side-v2-goal-observable', 'peg-unplug-side-v2-goal-observable', 'soccer-v2-goal-observable', 'stick-push-v2-goal-observable', 'stick-pull-v2-goal-observable', 'push-wall-v2-goal-observable', 'reach-wall-v2-goal-observable', 'shelf-place-v2-goal-observable', 'sweep-into-v2-goal-observable', 'sweep-v2-goal-observable', 'window-open-v2-goal-observable', 'window-close-v2-goal-observable']
def get_policy(env_name):
    name = "".join(" ".join(env_name.split('-')[:-3]).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy

def save_frame(path, frame):
    imageio.imwrite(path, frame)
    
ps = {}
for env_name in env_dict.keys():
    policy = get_policy(env_name)
    if policy is None:
        print("Policy not found:", env_name)
    else:
        ps[env_name] = policy

out_path = collection_config["output_path"]

os.makedirs(out_path, exist_ok=True)
for task in tqdm(included_tasks):
    print(task)
    for camera in collection_config["camera_names"]:
        demos = []
        action_seqs = []
        raw_lengths = []
        rewards = []
        for seed in tqdm(range(42, 42+ceil(collection_config["demos_per_camera"] * (1+collection_config["discard_ratio"])))):
            env = env_dict[task](seed=seed)
            obs = env.reset()
            images, _, action_seq, reward = collect_video(obs, env, ps[task], camera_name=camera, resolution=collection_config["resolution"])
            # assert len(images) == len(action_seq) + 1 or len(images) == 502
            raw_lengths += [len(images)]
            demos += [images]
            action_seqs += [action_seq]
            rewards += [reward]
        top_k_ind = np.argsort(raw_lengths)[:collection_config["demos_per_camera"]]
        demos = [demos[i] for i in top_k_ind]
        raw_lengths = [raw_lengths[i] for i in top_k_ind]
        action_seqs = [action_seqs[i] for i in top_k_ind]
        rewards = [rewards[i] for i in top_k_ind]
        print(f"vid length bounds: {raw_lengths[0]} ~ {raw_lengths[-1]}")
        
        ### save the collected demos
        out_dir = os.path.join(out_path, "-".join(task.split('-')[:-3]))
        os.makedirs(out_dir, exist_ok=True)
        for i, demo in enumerate(demos):
            demo_dir = os.path.join(out_dir, f"{camera}/{i:03d}")
            os.makedirs(demo_dir, exist_ok=True)
            with mp.Pool(10) as p:
                p.starmap(save_frame, [(os.path.join(demo_dir, f"{j:02d}.png"), frame) for j, frame in enumerate(demo)])
            with open(f"{demo_dir}/action.pkl", "wb") as f:
                pickle.dump(action_seqs[i], f)
            with open(f"{demo_dir}/rewards.pkl", "wb") as f:
                    pickle.dump(rewards[i], f)
        
        

        
    
    
        
        