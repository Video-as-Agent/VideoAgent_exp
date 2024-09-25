from .goal_diffusion import GoalGaussianDiffusion, Trainer
from .goal_diffusion_v1 import GoalGaussianDiffusion as GoalGaussianDiffusion_v1, Trainer as Trainer_v1
from .goal_diffusion_policy import GoalGaussianDiffusion as GoalGaussianDiffusionPolicy, Trainer as TrainerPolicy
from .diffusion_policy_baseline.unet import Unet1D, TransformerNet
from .unet import UnetMW as Unet
from .unet import UnetMWFlow as Unet_flow
from .unet import UnetThor as Unet_thor
from .unet import UnetBridge as Unet_bridge
from .feedback_binary_rf import chat_with_openai_rf
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import utils
from torchvision import transforms as T
from einops import rearrange
import torch
import imageio
from PIL import Image
from torch import nn
import numpy as np
import os

cache_dir = "../.cache/0001/"
os.makedirs(cache_dir, exist_ok=True)

def get_diffusion_policy_T(ckpt_dir='../ckpts/diffusion_policy_T', milestone=1, sampling_timesteps=10):
    unet = TransformerNet()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusionPolicy(
        channels=4,
        model=unet,
        image_size=10,
        timesteps=100,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    trainer = TrainerPolicy(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[0],
        valid_set=[0],
        train_lr=1e-4,
        train_num_steps =100000,
        save_and_sample_every =2500,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =32,
        valid_batch_size =1,
        gradient_accumulate_every = 1,
        num_samples=1, 
        results_folder =ckpt_dir,
        fp16 =True,
        amp=True,
    )

    trainer.load(milestone)
    return trainer

class DiffusionPolicy_T():
    def __init__(self, milestone=10, amp=True, sampling_timesteps=10):
        self.policy = get_diffusion_policy_T(milestone=milestone, sampling_timesteps=sampling_timesteps)
        self.amp = amp
        self.transform = T.Compose([
            T.Resize((320, 240)),
            T.CenterCrop((128, 128)),
            T.ToTensor(),
        ])

    def __call__(self,
                obs: np.array,
                task: str,
            ):
        device = self.policy.device
        obs = torch.stack([self.transform(Image.fromarray(o)) for o in obs], dim=0).float().to(device).unsqueeze(0)
        with torch.no_grad():
            return self.policy.sample(obs, [task]).cpu().squeeze(0).numpy()

def get_diffusion_policy(ckpt_dir='../ckpts/diffusion_policy', milestone=1, sampling_timesteps=10):
    unet = Unet1D()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusionPolicy(
        channels=4,
        model=unet,
        image_size=16,
        timesteps=100,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    trainer = TrainerPolicy(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[0],
        valid_set=[0],
        train_lr=1e-4,
        train_num_steps =100000,
        save_and_sample_every =2500,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =32,
        valid_batch_size =1,
        gradient_accumulate_every = 1,
        num_samples=1, 
        results_folder =ckpt_dir,
        fp16 =True,
        amp=True,
    )

    trainer.load(milestone)
    return trainer

class DiffusionPolicy():
    def __init__(self, milestone=10, amp=True, sampling_timesteps=10):
        self.policy = get_diffusion_policy(milestone=milestone, sampling_timesteps=sampling_timesteps)
        self.amp = amp
        self.transform = T.Compose([
            T.Resize((320, 240)),
            T.CenterCrop((128, 128)),
            T.ToTensor(),
        ])

    def __call__(self,
                obs: np.array,
                task: str,
            ):
        device = self.policy.device
        obs = torch.stack([self.transform(Image.fromarray(o)) for o in obs], dim=0).float().to(device).unsqueeze(0)
        with torch.no_grad():
            return self.policy.sample(obs, [task]).cpu().squeeze(0).numpy()


def get_video_model(ckpts_dir='../ckpts/metaworld', milestone=24, flow=False, timestep=100):
    unet = Unet_flow() if flow else Unet()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = 8
    target_size = (128, 128)
    channels = 3 if not flow else 2

    diffusion = GoalGaussianDiffusion(
        channels=channels*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=timestep,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    
    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load(milestone)
    return trainer
        
def get_video_model_thor(ckpts_dir='../ckpts/ithor', milestone=30):
    unet = Unet_thor()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = 8
    target_size = (64, 64)
    channels = 3

    diffusion = GoalGaussianDiffusion(
        channels=channels*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=100,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    
    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load(milestone)
    return trainer

def get_video_model_bridge(ckpts_dir='../ckpts/bridge', milestone=42):
    unet = Unet_bridge()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = 8
    target_size = (48, 64)
    channels = 3

    diffusion = GoalGaussianDiffusion_v1(
        model=unet,
        image_size=target_size,
        channels=channels*(sample_per_seq-1),
        timesteps=100,
        sampling_timesteps=100,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    
    trainer = Trainer_v1(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load(milestone)
    return trainer

def pred_video(model, frame_0, task, flow=False):
    device = model.device
    original_shape = frame_0.shape
    center = (original_shape[1]//2, original_shape[0]//2)
    xpad, ypad = center[0]-64, center[1]-64

    channels = 3 if not flow else 2
    
    transform = T.Compose([
        T.CenterCrop((128, 128)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]
    preds = rearrange(model.sample(image.to(device), text).cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    if not flow:
        preds = torch.cat([image, preds], dim=0)
    # pad the image back to original shape (both sides)
    images = torch.nn.functional.pad(preds, (xpad, xpad, ypad, ypad))
    return images.numpy().transpose(0, 2, 3, 1) * 128 if flow else (images.numpy()*255).astype('uint8')

def pred_video_thor(model, frame_0, task):
    channels=3
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]
    preds = rearrange(model.sample(image, text).cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    preds = torch.cat([image, preds], dim=0)
    return (preds.numpy()*255).astype('uint8')

def pred_video_thor_iterative(model, frame_0, task, model_type='vid', max_iterations = 4, eval_ = "VLM"):
    print("Model Type: ", model_type)
    print("Max_iterations: ", max_iterations)
    original_shape = frame_0.shape
    print("Original Shape: ", original_shape)
    iteration = 0
    channels=3
    target_size = (64, 64)
    vid_conditions = []
    outputs = []
    gif_paths = []
    
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]
    output = model.sample(image, text)
    preds = rearrange(output.cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    preds = torch.cat([image, preds], dim=0)
    vid_conditions.append((preds.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8'))
    gif_path = cache_dir + task + "-" + str(iteration) + "_out.gif"
    imageio.mimsave(gif_path, vid_conditions[0], duration=200, loop=1000)
    gif_paths.append(gif_path)
    
    image_path = cache_dir + task + ".png"
    imageio.imwrite(image_path, Image.fromarray(frame_0))
    
    outputs.append((preds.numpy()*255).astype('uint8'))
    if model_type == 'vid':
        print("Video Conditioned Generation")
        while iteration < max_iterations:
            if eval_ == "VLM":
                print("VLM_eval")
                cnt = 0
                # for i in range(5):
                response = chat_with_openai_rf([image_path],[gif_paths[-1]],[task])
                print(response)
                    #response[0] = 'Reject'
                if response[0][0] == 'A':
                #         cnt = cnt + 1
                # if cnt >= 3:       
                    print("Output from iteration: ", iteration)
                    return outputs[-1]
            iteration = iteration + 1
            print("iteration: ",iteration, "output_shape:", output.shape)
            output = rearrange(output, 'b (f c) w h -> b c f w h', c = 3)
            output = model.sample(output, text).cpu()
            output_1 = output[0].reshape(-1, 3, *target_size)
            output_1 = torch.cat([image, output_1], dim=0)
            
            gif_path = cache_dir + task + "-" + str(iteration) + "_out.gif"
            vid_conditions.append((output_1.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8'))
            
            # gif_path = cache_dir + task + "-" + str(iteration) + "_out.gif"
            # output_images = (output_1.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')
            # output_images = [np.array(Image.fromarray(frame).resize((320, 240))) for frame in output_images]
            # vid_conditions.append(output_images)
            
            imageio.mimsave(gif_path, vid_conditions[iteration], duration=200, loop=1000)
            gif_paths.append(gif_path)
            outputs.append((output_1.numpy()*255).astype('uint8'))
    print("Last Iteration")
    return outputs[1]


def pred_video_bridge(model, frame_0, task):
    channels=3
    transform = T.Compose([
        T.Resize((48, 64)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]
    preds = rearrange(model.sample(image, text).cpu().squeeze(0), "(f c) w h -> f c w h", c=channels)
    preds = torch.cat([image, preds], dim=0)
    return (preds.numpy()*255).astype('uint8')


def pred_video_iterative(model, model_type, frame_0, task, max_iterations = 3, camera = 'corner1', eval_ = "None",flow=False):
    print("Model Type: ", model_type)
    print("Max_iterations: ", max_iterations)
    
    device = model.device
    original_shape = frame_0.shape
    print("Original Shape: ", original_shape)
    center = (original_shape[1]//2, original_shape[0]//2)
    xpad, ypad = center[0]-64, center[1]-64
    iteration = 0
    target_size = (128, 128)

    vid_conditions = []
    outputs = []
    gif_paths = []

    channels = 3 if not flow else 2
    
    transform = T.Compose([
        T.CenterCrop((128, 128)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]

    output = model.sample(image.to(device), text)

    preds = rearrange(output.cpu().squeeze(0), "(f c) w h -> f c w h", c=channels) 
    if not flow:
        preds = torch.cat([image, preds], dim=0) #torch.Size([8, 3, 128, 128])
    # pad the image back to original shape (both sides)
    vid_conditions.append((preds.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8'))
    gif_path = cache_dir + task + "-" + str(iteration) + "_out.gif"
    imageio.mimsave(gif_path, vid_conditions[0], duration=200, loop=1000)
    gif_paths.append(gif_path)

    image_path = cache_dir + task + ".png"
    imageio.imwrite(image_path, Image.fromarray(frame_0))
    #save image at image_path
    
    images = torch.nn.functional.pad(preds, (xpad, xpad, ypad, ypad))
    flow_out = images.numpy().transpose(0, 2, 3, 1) * 128 if flow else (images.numpy()*255).astype('uint8')
    outputs.append(flow_out)

    

    if model_type == 'vid':
        print("Video Conditioned Generation")
        while iteration < max_iterations:
            if eval_ == "VLM":
                print("VLM_eval")
                # cnt = 0
                # for i in range(5):
                response = chat_with_openai_rf([image_path],[gif_paths[-1]],[task])
                print(response)
                    #response[0] = 'Reject'
                if response[0][0] == 'A':
                #         cnt = cnt + 1
                # if cnt >= 3:       
                    print("Output from iteration: ", iteration)
                    return outputs[-1]
            iteration = iteration + 1
            print("iteration: ",iteration, "output_shape:", output.shape)
            output = rearrange(output, 'b (f c) w h -> b c f w h', c = 3)
            output = model.sample(output, text).cpu()
            output_1 = output[0].reshape(-1, 3, *target_size)
            output_1 = torch.cat([image, output_1], dim=0) #torch.Size([8, 3, 128, 128])

            gif_path = cache_dir + task + "-" + str(iteration) + "_out.gif"
            vid_conditions.append((output_1.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8'))
            imageio.mimsave(gif_path, vid_conditions[iteration], duration=200, loop=1000)
            gif_paths.append(gif_path)
             
            images = torch.nn.functional.pad(output_1, (xpad, xpad, ypad, ypad))
            flow_out = images.numpy().transpose(0, 2, 3, 1) * 128 if flow else (images.numpy()*255).astype('uint8')
            outputs.append(flow_out)
            

    if model_type == 'binary':
        while iteration < max_iterations:
            iteration = iteration + 1
            response = chat_with_openai_rf([image_path],[gif_paths[iteration-1]],[task])
            text = [task + ", feedback is " + response[0]]

            output = rearrange(output, 'b (f c) w h -> b c f w h', c = 3)
            output = model.sample(output, text).cpu()
            output_1 = output[0].reshape(-1, 3, *target_size)
            output_1 = torch.cat([image, output_1], dim=0)

            gif_path = cache_dir + task + "-" + str(iteration) + "_out.gif"
            vid_conditions.append((output_1.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8'))
            imageio.mimsave(gif_path, vid_conditions[iteration], duration=200, loop=1000)
            gif_paths.append(gif_path)

            images = torch.nn.functional.pad(output_1.squeeze(0), (xpad, xpad, ypad, ypad))
            flow_out = images.numpy().transpose(0, 2, 3, 1) * 128 if flow else (images.numpy()*255).astype('uint8')
            outputs.append(flow_out)

    
    
    # if eval_ == "VLM":
    #     print("VLM_eval")
    #     for i in range(len(gif_paths)):
    #         response = chat_with_openai_rf([image_path],[gif_paths[i]],[task])
    #         print(response)
    #         #response[0] = 'Reject'
    #         if response[0][0] == 'A': 
    #             print("Output from iteration: ", i)
    #             return outputs[i]
    print("Last Iteration")
    return outputs[-1]


def pred_video_iterative_flow(model, model_type, frame_0, task, max_iterations = 3, camera = 'corner1', eval_ = "flow",flow=False):
    print("Model Type: ", model_type)
    print("Max_iterations: ", max_iterations)
    
    device = model.device
    original_shape = frame_0.shape
    center = (original_shape[1]//2, original_shape[0]//2)
    xpad, ypad = center[0]-64, center[1]-64
    iteration = 0
    target_size = (128, 128)

    vid_conditions = []
    outputs = []
    gif_paths = []

    channels = 3 if not flow else 2
    
    transform = T.Compose([
        T.CenterCrop((128, 128)),
        T.ToTensor(),   
    ])
    image = transform(Image.fromarray(frame_0)).unsqueeze(0)
    text = [task]

    output = model.sample(image.to(device), text)

    preds = rearrange(output.cpu().squeeze(0), "(f c) w h -> f c w h", c=channels) 
    if not flow:
        preds = torch.cat([image, preds], dim=0)
    # pad the image back to original shape (both sides)
    vid_conditions.append((preds.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8'))
    gif_path = cache_dir + task + "-" + str(iteration) + "_out.gif"
    imageio.mimsave(gif_path, vid_conditions[0], duration=200, loop=1000)
    gif_paths.append(gif_path)

    image_path = cache_dir + task + ".png"
    imageio.imwrite(image_path, Image.fromarray(frame_0))
    #save image at image_path

    images = torch.nn.functional.pad(preds, (xpad, xpad, ypad, ypad))
    flow_out = images.numpy().transpose(0, 2, 3, 1) * 128 if flow else (images.numpy()*255).astype('uint8')
    outputs.append(flow_out)

    if model_type == 'vid':
        print("Video Conditioned Generation")
        while iteration < max_iterations:
            iteration = iteration + 1
            print(output.shape)
            output = rearrange(output, 'b (f c) w h -> b c f w h', c = 3)
            output = model.sample(output, text).cpu()
            output_1 = output[0].reshape(-1, 3, *target_size)
            output_1 = torch.cat([image, output_1], dim=0)

            gif_path = cache_dir + task + "-" + str(iteration) + "_out.gif"
            vid_conditions.append((output_1.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8'))
            imageio.mimsave(gif_path, vid_conditions[iteration], duration=200, loop=1000)
            gif_paths.append(gif_path)

            images = torch.nn.functional.pad(output_1.squeeze(0), (xpad, xpad, ypad, ypad))
            flow_out = images.numpy().transpose(0, 2, 3, 1) * 128 if flow else (images.numpy()*255).astype('uint8')
            outputs.append(flow_out)

    if model_type == 'binary':
        while iteration < max_iterations:
            iteration = iteration + 1
            response = chat_with_openai_rf([image_path],[gif_paths[iteration-1]],[task])
            text = [task + ", feedback is " + response[0]]

            output = rearrange(output, 'b (f c) w h -> b c f w h', c = 3)
            output = model.sample(output, text).cpu()
            output_1 = output[0].reshape(-1, 3, *target_size)
            output_1 = torch.cat([image, output_1], dim=0)

            gif_path = cache_dir + task + "-" + str(iteration) + "_out.gif"
            vid_conditions.append((output_1.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8'))
            imageio.mimsave(gif_path, vid_conditions[iteration], duration=200, loop=1000)
            gif_paths.append(gif_path)

            images = torch.nn.functional.pad(output_1.squeeze(0), (xpad, xpad, ypad, ypad))
            flow_out = images.numpy().transpose(0, 2, 3, 1) * 128 if flow else (images.numpy()*255).astype('uint8')
            outputs.append(flow_out)

    if model_type == 'base_re':
        while iteration < max_iterations:
            iteration = iteration + 1
            output = model.sample(image.to(device), text)

            preds = rearrange(output.cpu().squeeze(0), "(f c) w h -> f c w h", c=channels) 
            if not flow:
                preds = torch.cat([image, preds], dim=0)
            # pad the image back to original shape (both sides)
            vid_conditions.append((preds.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8'))
            gif_path = cache_dir + task + "-" + str(iteration) + "_out.gif"
            imageio.mimsave(gif_path, vid_conditions[0], duration=200, loop=1000)
            gif_paths.append(gif_path)

            image_path = cache_dir + task + ".png"
            imageio.imwrite(image_path, Image.fromarray(frame_0))
            #save image at image_path

            images = torch.nn.functional.pad(preds, (xpad, xpad, ypad, ypad))
            flow_out = images.numpy().transpose(0, 2, 3, 1) * 128 if flow else (images.numpy()*255).astype('uint8')
            outputs.append(flow_out)

    # if camera == 'corner3': return outputs[0]
    
    # if eval_ == "VLM":
    #     print("VLM_eval")
    #     for i in range(len(gif_paths)):
    #         response = chat_with_openai_rf([image_path],[gif_paths[i]],[task])
    #         print(response)
    #         #response[0] = 'Reject'
    #         if response[0][0] == 'A': 
    #             print("Output")
    #             return outputs[i]
    print("Done Outputs Given")
    return outputs
            