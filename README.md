## open-LiteVAE

---
[![arxiv](https://img.shields.io/badge/arXiv-2405.14477-red)](https://arxiv.org/abs/2405.14477)
 
Implementation of **"[LiteVAE: Lightweight and Efficient Variational Autoencoders for Latent Diffusion Models](https://openreview.net/forum?id=mTAbl8kUzq)"** [2024]. The paper introduces an efficient wavelet-encoder-based variational autoencoder, which demonstrates a significant performance improvement and stable training compared with previous works. This implementation aims to replicate and extend its findings using GPU-accelerated wavelet transformations (**[torch-dwt](https://github.com/KeKsBoTer/torch-dwt)**), stochastic image rescaling, and improved discriminator models.

#### Note
This implementation was independently developed **before the authors provided pseudocode** in the appendix of their paper. As a result, the approach here may differ slightly in details but adheres to the paper's methodology and goals.

---
## Comparisons 

#### Model Configurations

Comparison of model configurations for n<sub>z</sub>=12.

<table>
  <thead>
    <tr>
      <th rowspan="2">Scale</th>
      <th rowspan="1" colspan="2">Parameter Count</th>
      <th colspan="3">Extractor</th>
      <th colspan="3">Aggregator</th>
    </tr>
    <tr>
    <th>LiteVAE</th>
     <th> Ours </th>
      <th>C</th>
      <th>Mult</th>
      <th>Blocks</th>
      <th>C</th>
      <th>Mult</th>
      <th>Blocks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>S</td>
      <td>1.03 M</td>
      <td>0.97 M</td>
      <td>16</td>
      <td>[1,2,2]</td>
      <td>3</td>
      <td>16</td>
      <td>[1,2,2]</td>
      <td>3</td>
    </tr>
    <tr>
      <td>B</td>
      <td>6.75 M</td>
      <td>6.6 M</td>
      <td>32</td>
      <td>[1,2,3]</td>
      <td>4</td>
      <td>32</td>
      <td>[1,2,3]</td>
      <td>4</td>
    </tr>
        <tr>
      <td>M</td>
      <td>32.75 M</td>
      <td>34 M </td>
      <td>64</td>
      <td>[1,2,4]</td>
      <td>5</td>
      <td>32</td>
      <td>[1,2,3]</td>
      <td>4</td>
    </tr>
      <tr>
      <td>L</td>
      <td>41.42 M</td>
      <td>41.15 M</td>
      <td>64</td>
      <td>[1,2,4]</td>
      <td>5</td>
      <td>64</td>
      <td>[1,2,4]</td>
      <td>4</td>
    </tr>
  </tbody>
</table>

#### Evaluation Metrics

Evaluation metrics were computed on the ImageNet training set with the B-Scale encoder using n<sub>z</sub>=12. Training is conducted in two phases: A) pre-training at 128x128 with no discriminator for 100k steps, B) finetuning at 256x256 with discriminator for 50k steps. 

All metrics are computed on the full ImageNet-1k validation set (50k images) using bi-cubic rescaling and center cropping.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2"> Disc Type </th>
      <th colspan="3">Loss Weights</th>
      <th colspan="4">Evaluation 128x128</th>
      <th colspan="4">Evaluation 256x256</th>
    </tr>
    <tr>
      <th>w<sub>kl</sub></th>
      <th>w<sub>wave</sub></th>
      <th>w<sub>gauss</sub></th>
      <th>LPIPS</th>
      <th>PSNR</th>
      <th>rFID</th>
      <th>SSIM</th>
      <th>LPIPS</th>
      <th>PSNR</th>
      <th>rFID</th>
      <th>SSIM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VAE</td>
      <td>Stem</td>
      <!-- Weights -->
      <td>?</td>
      <td>N/A</td>
      <td>N/A</td>
      <!-- 128x128 -->
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <!-- 256x256 -->
      <td>0.069</td>
      <td>29.25</td>
      <td>0.95</td>
      <td>0.86</td>
    </tr>
     <tr>
      <td>LiteVAE</td>
      <td>Stem?</td>
      <!-- Weights -->
      <td>?</td>
      <td>?</td>
      <td>?</td>
      <!-- 128x128 -->
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <td>--</td>
      <!-- 256x256 -->
      <td>0.069</td>
      <td>29.55</td>
      <td>0.94</td>
      <td>0.87</td>
    </tr>
    <tr>
      <td>A1 (Ours)</td>
      <td>N/A</td>
      <!-- Weights -->
      <td>0.01</td>
      <td>0.1</td>
      <td>0.1</td>
      <!-- 128x128 -->
      <td>0.077</td>
      <td>28.35</td>
      <td>3.78</td>
      <td>0.85</td>
      <!-- 256x256 -->
      <td>0.081</td>
      <td>29.38</td>
      <td>1.86</td>
      <td>0.85</td>
    </tr>
    <tr>
      <td>A2 (Ours)</td>
      <td>N/A</td>
      <!-- Weights -->
      <td>0.01</td>
      <td>1.0</td>
      <td>0.5</td>
      <!-- 128x128 -->
      <td>0.080</td>
      <td>28.76</td>
      <td>3.37</td>
      <td>0.86</td>
      <!-- 256x256 -->
      <td>0.084</td>
      <td>29.78</td>
      <td>1.12</td>
      <td>0.86</td>
    </tr>
  </tbody>
</table>

---


## Prerequisites
- Python >= 3.9
- PyTorch >= 2.0
- Torch-DWT


---

## TODO

- [ ] Add Description of Improved Methods
- [ ] Add Training Code
- [ ] Add Evaluation Code
- [ ] More Experiments

---


## Please Cite the Original Paper

```
@inproceedings{
sadat2024litevae,
title={Lite{VAE}: Lightweight and Efficient Variational Autoencoders for Latent Diffusion Models},
author={Seyedmorteza Sadat and Jakob Buhmann and Derek Bradley and Otmar Hilliges and Romann M. Weber},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=mTAbl8kUzq}
}
```

