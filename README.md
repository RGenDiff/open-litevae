## open-LiteVAE

---
[![arxiv](https://img.shields.io/badge/arXiv-2405.14477-red)](https://arxiv.org/abs/2405.14477)
 
Implementation of **"[LiteVAE: Lightweight and Efficient Variational Autoencoders for Latent Diffusion Models](https://openreview.net/forum?id=mTAbl8kUzq)"** [2024]. The paper introduces an efficient wavelet-encoder-based variational autoencoder, which demonstrates a significant performance improvement and stable training compared with previous works. This implementation aims to replicate and extend its findings using GPU-accelerated wavelet transformations (**[torch-dwt](https://github.com/KeKsBoTer/torch-dwt)**), stochastic image rescaling, and improved discriminator models.

#### Note
This implementation was independently developed **before the authors provided pseudocode** in the appendix of their paper. As a result, the approach here may differ slightly in details but adheres to the paper's methodology and goals.


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

---
## Model Configurations 

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


Comparison of Discriminators 

<table>
<thead>
	 <tr>
     <th> Model </th>
     <th> Params </th>
     <th> FLOPs </th>
     <th> Config (256x256) </th>
     </tr>
</thead>
<tbody>
	<tr>
    <td> PatchGAN </td>
    <td> 2.77M </td>
    <td> 3.15G </td>
    <td> n<sub>layers</sub>=3, n<sub>df</sub>=64 </td>
    </tr>
	<tr>
    <td> GigaGAN </td>
    <td> 14.38M </td>
    <td> 3.23G </td>
    <td> C<sub>base</sub>=4096, C<sub>max</sub>=256, n<sub>blocks</sub>=2, attn=[8,16] </td>
    </tr>
	<tr>
    <td> UNetGAN-S </td>
    <td> 2.75M  </td>
    <td> 2.31G </td>
    <td> D<sub>ch</sub>=16, attn=None </td>
    </tr>
	<tr>
    <td> UNetGAN-M </td>
    <td> 11.0M </td>
    <td> 9.13G </td>
    <td> D<sub>ch</sub>=32, attn=None </td>
    </tr>
</tbody>
</table>

---

## Comparisons

#### Evaluation Metrics

Evaluation metrics were computed on the ImageNet training set with the B-Scale encoder using n<sub>z</sub>=12. Training is conducted in two phases: A) pre-training at 128x128 with no discriminator for 100k steps, B) finetuning at 256x256 with discriminator for 50k steps. 

All metrics are computed on the full ImageNet-1k validation set (50k images) using bi-cubic rescaling and center cropping.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2"> Disc Type </th>
      <th colspan="3">Loss Weights</th>
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
      <!-- 256x256 -->
      <td>0.084</td>
      <td>29.78</td>
      <td>1.12</td>
      <td>0.86</td>
    </tr>
  </tbody>
</table>


---

## Repository Structure

```
configs/         
    - Configuration files for setting up the experiments and models.

olvae/
    - Main directory for the Open LiteVAE project.
    ├── data/        
    │   - Contains the dataloader for data preparation and augmentation. 
    ├── models/      
    │   - Contains PyTorch Lightning models for training and evaluation. 
    ├── modules/     
        - Contains model-specific layers and architectures.
        ├── litevae/
        │   - Specialized layers for the LiteVAE model. 
        ├── basicgan/
        │   - Common GAN components and loss functions [include PatchGAN discriminator]. 
        ├── gigagan/
        │   - Layers specific to the GigaGAN discriminator. 
        ├── unetgan/
        │   - Layers and components for the UnetGAN discriminator. 

scripts/
    - Code for evaluation, testing, and utilities.
```

---

## Prerequisites
- Python >= 3.9
- PyTorch >= 2.0
- Torch-DWT


---

## TODO

- [ ] Add Setup Documentation
- [ ] Add Description of Improved Methods
- [ ] Add Training Code
- [ ] Add Evaluation Code
- [ ] More Experiments

---

## References 




