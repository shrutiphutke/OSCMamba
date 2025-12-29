# OSCMamba: Omni-directional Selective Scan Convolution Mamba for Medical Image Classification
This is the official code repository for "***OSCMamba: Omni-directional Selective Scan Convolution Mamba for Medical Image Classification***"[![Springer](https://link.springer.com/chapter/10.1007/978-3-031-93709-5_33)
# 📝Abstract📝
The advancement of various learning approaches has a great impact in computer vision applications specifically for medical image analysis. Being the most important task, classification accuracy of medical images has been successively improved using different methods such as Convolutional neural networks (CNNs), Transformers, etc. These models have some limitations such as the CNNs perform poorly when the feature extraction considering long-range dependency is concerned. Whereas Transformers perform well while dealing with the long-range dependencies for feature extraction, leading to the quadratic complexity. The evolution of state space models (SSMs) deals with the limitations of both the CNNsand Transformers. This has the advent of capturing the long range dependencies and the linear complexity. Further, the scanning mechanism in the SSM provides the advantage of focusing on the required features while ignoring the rest. The existing Mamba based approach for medical image classification considers only horizontal and vertical feature scanning ignoring the diagonal information. While the omni-directional selective scan considers all of them. With this motivation, we propose omni directional selective scan-based convolution mamba (OSCMamba)
approach for medical image classification. The OSCMamba approach is applied on different medical image modalities for image classification. The detailed experimental analysis with AUC and ACC on six different datasets proves the efficiency of the proposed OSCMamba basedmmedical image classification approach.

# 📌Installation📌
* `pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117`
* `pip install einops`
* `pip install packaging`
* `pip install timm==0.4.12`
* `pip install pytest chardet yacs termcolor`
* `pip install submitit tensorboardX`
* `pip install triton==2.0.0`
* `pip install causal_conv1d>=1.4.0`
* `pip install mamba_ssm`
* `pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs`
## 📜Other requirements📜:
* Linux System
* NVIDIA GPU
* CUDA 12.0+
# The classification performance of OSCMamba
The classification performance of OSCMamba is evaluated on six different datasets.
![dataset_01](https://github.com/shrutiphutke/OSCMamba/blob/main/dataset.png)

# 💞Citation💞
If you find this repository useful, please consider the following references. We would greatly appreciate it.
```bibtex
@inproceedings{phutke2024oscmamba,
  title={OSCMamba: Omni-Directional Selective Scan Convolution Mamba for Medical Image Classification},
  author={Phutke, Shruti and Shakya, Amit and Gupta, Chetan and Kumar, Rupesh and Sharma, Lalit},
  booktitle={International Conference on Computer Vision and Image Processing},
  pages={461--475},
  year={2024},
  organization={Springer}
}
```
# ✨Acknowledgments✨
We thank the authors of [MedMamba](https://github.com/YubiaoYue/MedMamba) for their open-source code.
