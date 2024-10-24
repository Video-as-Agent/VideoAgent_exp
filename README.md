# VideoAgent experiments

The official codebase for running the experiments described in the VideoAgent paper. You can find codebase for training video policies [here](https://github.com/Video-as-Agent/VideoAgent.git).

This repository contains the code for training video policies presented in our work   
[VideoAgent: Self improving video generation](https://arxiv.org/pdf/2410.10076)  
[Achint Soni](https://trickyjustice.github.io),
[Sreyas Venkataraman](https://github.com/vsreyas),
[Abhranil Chandra](https://abhranilchandra.github.io),
[Sebastian Fischmeister](https://uwaterloo.ca/embedded-software-group/profiles/sebastian-fischmeister),
[Percy Liang](https://cs.stanford.edu/~pliang/),
[Bo Dai](https://bo-dai.github.io),
[Sherry Yang](https://sherryy.github.io)
[website](https://video-as-agent.github.io) | [paper](https://arxiv.org/pdf/2410.10076) | [arXiv](https://arxiv.org/abs/2410.10076) | [experiment repo](https://github.com/Video-as-Agent/VideoAgent_exp)

```bib
@misc{soni2024videoagentselfimprovingvideogeneration,
      title={VideoAgent: Self-Improving Video Generation}, 
      author={Achint Soni and Sreyas Venkataraman and Abhranil Chandra and Sebastian Fischmeister and Percy Liang and Bo Dai and Sherry Yang},
      year={2024},
      eprint={2410.10076},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.10076}, 
}
```


## Getting started

We recommend to create a new environment with pytorch installed using conda. 

```bash  
conda create -n videoagent_exp python=3.9
conda activate videoagent_exp
conda install pytorch=2.2.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```  

Next, clone the repository and install the requirements  

```bash
git clone https://github.com/Video-as-Agent/VideoAgent_exp.git
cd VideoAgent_exp
pip install -r requirements.txt
```

## Download the Checkpoints

We provide the checkpoints used in our main experiments. You can download them using `download.sh`, for example:

```bash
bash download.sh metaworld
# bash download.sh metaworld-DA
# bash download.sh ithor 
```

## Running the Experiments

First, cd into the `experiment` directory. 

```bash
cd experiment
```

### Meta-World

To run the full VideoAgent on Meta-World, run the following command:

```bash
# make sure you have the checkpoint ../ckpts/metaworld/model-24.pt
bash benchmark_mw.sh 0
# the argument 0 is the GPU id, you can change it to other GPU id if you wish
```

We have provided also provided another checkpoint trained with simple random-shift data augmentation. Specifically we first center cropped the image to 160x160 from the original 320x240 image and then random-crop an 128x128 image from it. We found slightly improved performance with this simple augmentation. 

To run the full VideoAgent on Meta-World with this checkpoint, run the following command:

```bash
# make sure you have the checkpoint ../ckpts/metaworld_DA/model-24.pt
bash benchmark_mw_DA.sh 0
```
To generate metaworld data for experiments, run the following command:

```bash
# make sure you change the collection config in collect_dataset_mw.py
python experiment/collect_dataset_mw.py
```


### iTHOR

To run the full VideoAgent on iTHOR, run the following command:

```bash
# make sure you have the checkpoint ../ckpts/ithor/model-24.pt
bash benchmark_thor.sh 0
```

## Acknowledgements

This codebase is modified from the following repositories:  
[AVDC](https://github.com/flow-diffusion/AVDC_experiments)


