from mypolicy import MyPolicy_CL, MyPolicy_CL_vids, MyPolicy_CL_fb
from metaworld_exp.utils import get_seg, get_cmat, collect_video, sample_n_frames
import sys
import pickle as pkl
sys.path.append('core')
import imageio.v2 as imageio
import numpy as np
from myutils import get_flow_model, pred_flow_frame, get_transforms, get_transformation_matrix
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from metaworld import policies
from tqdm import tqdm
import cv2
import imageio
import json
import os
from flowdiffusion.inference_utils import get_video_model, pred_video
import random
import torch
from argparse import ArgumentParser
import traceback
from torchvision import transforms
from PIL import Image
import imageio
import re
from glob import glob
from einops import rearrange
import time
import datetime
import dateutil.tz
import wandb


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def numerical_sort(value):
    """ Helper function to extract numbers for sorting filenames numerically. """
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

def get_task_text(env_name):
    name = " ".join(env_name.split('-')[:-3])
    return name

def get_policy(env_name):
    name = "".join(" ".join(get_task_text(env_name)).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy

with open("name2maskid.json", "r") as f:
    name2maskid = json.load(f)

def build_transforms( randomcrop, target_size):
        custom_crop = transforms.Lambda(lambda img: img.crop((80, 0, 280, 160)))
        if randomcrop:
            return transforms.Compose([
                custom_crop,
                transforms.CenterCrop((160, 160)),
                transforms.RandomCrop((128, 128)),
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                custom_crop,
                transforms.CenterCrop((128, 128)),
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])

def run(args):
    """h
    the sample returns video and image processed as we would feed the model
    x, x_2 = x.to(device), x_2.to(device)
    video_tensor = torch.cat([image, video_tensor], dim=0)
     images = torch.nn.functional.pad(video_tensor, (xpad, xpad, ypad, ypad))
    flow_out = images.numpy().transpose(0, 2, 3, 1) * 128 if flow else (images.numpy()*255).astype('uint8')
    I need trajectory ka seed, and then camera, along with the sampled traj
    
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    exp_name = "sim-feedback-base"
    exp_count = 0
    timestamp = now.strftime('%m_%d_%H_%M_%S')
    exp_name = "{}-{}-{:03}".format(exp_name, timestamp, exp_count)
    wandb.init(project="SI-Gen-Sim", name=exp_name, config=args)

    result_root = args.result_root
    os.makedirs(result_root, exist_ok=True)

    resolution = (320, 240)
    cameras = ['corner', 'corner2', 'corner3']
    max_replans = 0
    target_size=(128, 128)
    randomcrop = False
    device = "cuda"
    original_shape = (240, 320, 3)
    center = (original_shape[1]//2, original_shape[0]//2)
    xpad, ypad = center[0]-64, center[1]-64
    channels = 3
    
    base_path = '/home/ubuntu/achint/datasets/metaworld_split/metaworld_dataset'
    video_model = get_video_model(ckpts_dir=args.ckpt_dir, milestone=args.milestone)
    flow_model = get_flow_model()
    transform = build_transforms(randomcrop, target_size)
    base_actions_path = '/home/ubuntu/sreyas/dataset/metaworld/metaworld_dataset'
    
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            if "trajectory" in dir:
                env_name = root.split("/")[-3] + "-v2-goal-observable"
                seg_ids = name2maskid[env_name]
                benchmark_env = env_dict[env_name]
                
                print("Processing trajectory: ", dir)
                print("Env name: ", env_name)
                
                seed =  int(root.split("/")[-1]) + 500
                camera =  root.split("/")[-2]
                trajectory_dir = os.path.join(root, dir)
                print("Trajectory dir: ", trajectory_dir)
                feedback_file_path = os.path.join(trajectory_dir, 'feedback_sim.txt')
                return_file_path = os.path.join(trajectory_dir, 'return_sim.txt')
                
                image_files = sorted([f for f in os.listdir(trajectory_dir) if f.endswith('.png')], key=numerical_sort)
                video_files = sorted([f for f in os.listdir(trajectory_dir) if f.endswith('_output.gif')])
                print("image path: ", image_files[0])
                print("video path: ", video_files[0])
                
                if image_files and video_files:
                    
                    img_path = os.path.join(trajectory_dir, image_files[0])
                    video_path = os.path.join(trajectory_dir, video_files[0])
                    
                    img = Image.open(img_path)
                    images_tensor = transform(img).unsqueeze(0).to(device)
                    
                    video = imageio.mimread(video_path)
                    print("Number of frames:", len(video))
                    video_tensor = torch.stack([
                        transform(Image.fromarray(frame))  # Create a PIL image from each numpy array and apply the transform
                        for frame in video  # Process all frames
                    ]).to(device)
                    video_tensor = video_tensor.float() / 255.0
                    
                    # if video_tensor.dim() == 5:
                    #     video_tensor = rearrange(video_tensor.cpu().squeeze(0), "(f c) w h -> f c w h", c=channels) 
                    # else:
                    #     video_tensor = rearrange(video_tensor.cpu(), "(f c) w h -> f c w h", c=channels) 
                        # print("vid cond shape", x_2.shape)
                    video_tensor = torch.cat([images_tensor, video_tensor], dim=0)
                    images = torch.nn.functional.pad(video_tensor, (xpad, xpad, ypad, ypad))
                    flow_out =  (images.cpu().numpy()*255).astype('uint8')
                    
                    env = benchmark_env(seed=seed)
                    step = int(video_files[0].split("_")[0])
                    obs = env.reset()
                    
                    actions_path = os.path.join(base_actions_path, root.split("/")[-3], camera, root.split("/")[-1] ,"action.pkl")
                    with open(actions_path, 'rb') as f:
                        actions = pkl.load(f)
                    
                    i = 0
                    while i < step:
                        obs, _, _, _ = env.step(actions[i])
                        i += 1
                    
                    policy = MyPolicy_CL_fb(env, env_name, camera, video_model, flow_model, 'vid', flow_out,max_iterations = 0, max_replans=max_replans)

                    # os.makedirs(f'{result_root}/plans/{env_name}', exist_ok=True)
                    # imageio.mimsave(f'{result_root}/plans/{env_name}/{camera}_{seed}.mp4', images.transpose(0, 2, 3, 1))

                    images, _, episode_return = collect_video(obs, env, policy, camera_name=camera, resolution=resolution)
                    return_ = episode_return
                    
                    ### save sample video
                    os.makedirs(f'{result_root}/videos/{env_name}/{camera}/{seed}/{dir}', exist_ok=True)
                    imageio.mimsave(f'{result_root}/videos/{env_name}/{camera}/{seed}/{dir}/{step}_output_sim.mp4', images)
                    
                    print("test eplen: ", len(images))
                    if len(images) <= (500-step):
                        feedback = "Accept"
                        print("success, ",  "return: ", return_)
                    else:
                        feedback = "Reject"
                        print("failure,", "return: ", return_)
                        
                
                
                    """
                    Store in seq dataset, and write as a file in the particular location
                    """
                    formatted_feedback = f"{env_name}, {feedback}"
                    formatted_return = f"{env_name}, {return_}"
                    
                    with open(return_file_path, 'a') as return_file:
                        return_file.write('\n' + "base_video:" + formatted_return)
                    
                    with open(feedback_file_path, 'a') as feedback_file:
                            feedback_file.write('\n' + "base_video:" + formatted_feedback)
                    
                
                

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts/metaworld")
    parser.add_argument("--milestone", type=int, default=276)
    parser.add_argument("--result_root", type=str, default="../results/results_AVDC_full")
    args = parser.parse_args()
    run(args)
    
        