## BadSAM: Exploring Security Vulnerabilities of SAM via Backdoor Attacks


## Environment
This code was implemented with Python 3.8 and PyTorch 1.13.0. You can install all the requirements via:
```bash
pip install -r requirements.txt
```


## Quick Start
1. Download the dataset and put it in ./load.
2. Download the pre-trained [SAM(Segment Anything)](https://github.com/facebookresearch/segment-anything) and put it in ./pretrained.
3. Training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 loadddptrain.py --config configs/demo.yaml
```

4. Evaluation:
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch train.py --nnodes 1 --nproc_per_node 4 --config [CONFIG_PATH]
```

## Test
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Dataset

### Camouflaged Object Detection
- **[COD10K](https://github.com/DengPingFan/SINet/)**
- **[CAMO](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)**
- **[CHAMELEON](https://www.polsl.pl/rau6/datasets/)**

### Shadow Detection
- **[ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN)**


## Citation

If you find our work useful in your research, please consider citing:

```
@article{Guan_Hu_Zhou_Zhang_Li_Liu_2024, 
title={BadSAM: Exploring Security Vulnerabilities of SAM via Backdoor Attacks (Student Abstract)}, 
volume={38}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/30448}, 
DOI={10.1609/aaai.v38i21.30448}, 
number={21}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Guan, Zihan and Hu, Mengxuan and Zhou, Zhongliang and Zhang, Jielu and Li, Sheng and Liu, Ninghao}, 
year={2024}, 
month={Mar.}, 
pages={23506-23507}}
```

## Acknowledgements
The part of the code is derived from Explicit Visual Prompt   <a href='https://nifangbaage.github.io/Explicit-Visual-Prompt/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> by 
Weihuang Liu, [Xi Shen](https://xishen0220.github.io/), [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/), and [Xiaodong Cun](https://vinthony.github.io/) by University of Macau and Tencent AI Lab.

