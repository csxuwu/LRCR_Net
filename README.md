
# <p align=center> :fire: `Coarse-to-Fine Low-light Image Enhancement with Light Restoration and Color Refinement`</p>

![Python 3.8](https://img.shields.io/badge/python-3.8-g) ![pytorch 1.12.0](https://img.shields.io/badge/pytorch-1.12.0-blue.svg)

This is the official PyTorch codes for the paper.  
>**Coarse-to-Fine Low-light Image Enhancement with Light Restoration and Color Refinement**<br>  [Xu Wu](https://csxuwu.github.io/), Zhihui Lai<sup>*</sup>, Shiqi Yu, Jie Zhou, Zhuoqian Liang, Linlin Shen （ * indicates corresponding author)<br>
>IEEE Transactions on Emerging Topics in Computational Intelligence (TETCI), 2023


![framework_img](figs/LRCR_framework.png)

### Abstract

Low-light image enhancement aims to improve the illumination intensity while restoring color information. Despite recent advancements using deep learning methods, they still struggle with over or under-exposure in complex lighting scenes and poor color recovery in dark regions. To address these drawbacks, we propose a novel pipeline (called LRCR-Net) to perform Light Restoration and Color Refinement in a coarse-to-fine manner. In the coarse step, we focus on improving the illumination adaptively while avoiding inappropriate enhancement in the brighter or darker regions. This is achieved through introduce a region-calibrated residual block (RCRB) that balances local and global dependencies among different image regions. In the fine step, we aim to retouch the color of the images enhanced in the coarse step. To achieve this goal, we propose learnable image processing operators (LIPOs), including contrast and saturation operators, to refine the color according to the input images’ color and contrast information. The final result is an image with proper illumination and rich color. Experiments on four benchmark datasets (NASA, LIME, MEF, and NPE) show that our model outperforms state-of-the-art methods.

### :rocket: Highlights:
- A novel framework called LRCR-Net, whose most notable property is its progressive coarse-to-fine paradigm, is proposed for low-light image enhancement. With this advanced paradigm that overcomes the crucial challenges, LRCR-Net can achieve proper light restoration and visual pleasing color.
- In the coarse step, the RCRB is designed to fully explore and exploit low-light region features from both local and global perspectives. Moreover, the LIPOs are introduced in the fine step to refine the color output of the coarse step, thereby further improving the performance of the LRCR-Net.
- The effectiveness of LRCR-Net has been validated on various low-light enhancement benchmarks. These results demonstrate the superiority of our LRCR-Net over other state-of-the-art techniques.


### The whole process of image enhancement

<img src="figs/flow.png" width="800px">

## Experiments:
### NASA
<img src="figs/NASA.png" width="800px">

### LIME
<img src="figs/LIME.png" width="800px">

### NPE
<img src="figs/NPE.png" width="800px">

### Quantitative comparisons
<img src="figs/quantic_compare.png" width="800px">

## Dependencies and Installation

- CUDA >= 11.0
- Other required packages in `requirements.txt`

### Test LRCR
python test_for_LRCR.py

### Train LRCR
python trains_for_LRCR.py

## Citation
If you find our repo useful for your research, please cite us:
```
@inproceedings{wu2023ridcp,
    title={Coarse-to-Fine Low-light Image Enhancement with Light Restoration and Color Refinement},
    author={Xu Wu, Zhihui Lai, Shiqi Yu, Jie Zhou, Zhuoqian Liang, Linlin Shen},
    booktitle={IEEE Transactions on Emerging Topics in Computational Intelligence},
    year={2023}
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

## Acknowledgement
This repository is maintained by [Xu Wu](https://csxuwu.github.io/).
