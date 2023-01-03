# $D^4NeRF$

PyTorch implementation of paper [Detachable Novel Views Synthesis of Dynamic Scenes Using Distribution-Driven Neural Radiance Fields](http://arxiv.org/abs/2301.00411).

<div align=center>

<img src="https://github.com/Luciferbobo/D4NeRF/blob/main/demo/main.png" width="680"> 

</div>

<div align=center>
<img src="https://github.com/Luciferbobo/D4NeRF/blob/main/demo/S4.gif" width="210"> 
<img src="https://github.com/Luciferbobo/D4NeRF/blob/main/demo/Kid.gif" width="233"> 
<img src="https://github.com/Luciferbobo/D4NeRF/blob/main/demo/S2.gif" width="210"> 

**Novel View Synthesis**

</div>

## Abstract

Representing and synthesizing novel views in real-world dynamic scenes from casual monocular videos is a long-standing problem. Existing solutions typically approach dynamic scenes by applying geometry techniques or utilizing temporal information between several adjacent frames without considering the underlying background distribution in the entire scene or the transmittance over the ray dimension, limiting their performance on static and occlusion areas. Our approach **D**istribution-**D**riven neural radiance fields offers high-quality view synthesis and a 3D solution to **D**etach the background from the entire **D**ynamic scene, which is called $D^4$ NeRF. Specifically, it employs a neural representation to capture the scene distribution in the static background and a 6D-input NeRF to represent dynamic objects, respectively. Each ray sample is given an additional occlusion weight to indicate the transmittance lying in the static and dynamic components. We evaluate $D^4$ NeRF on public dynamic scenes and our urban driving scenes acquired from an autonomous-driving dataset. Extensive experiments demonstrate that our approach outperforms previous methods in rendering texture details and motion areas while also producing a clean static background. 

## Rendering Results

<div align=center>
  
<img src="https://github.com/Luciferbobo/D4NeRF/blob/main/demo/fig1.png" width="680"> 
  
**Comparison of novel view synthesis**

<img src="https://github.com/Luciferbobo/D4NeRF/blob/main/demo/fig3.png" width="680"> 
  
**Quantitative comparison results on NVIDIA dynamic scenes**
  
</div>

## Disentangled Results

<div align=center>

<img src="https://github.com/Luciferbobo/D4NeRF/blob/main/demo/fig2.png" width="680"> 
  
**Comparison on disentangled static background from entire scenes**
  
</div>

<div align=center>
  
<img src="https://github.com/Luciferbobo/D4NeRF/blob/main/demo/scene_ours.gif" width="200"> 
<img src="https://github.com/Luciferbobo/D4NeRF/blob/main/demo/scenedepth_ours.gif" width="200"> 
  
**The Entire Scene**

  
<img src="https://github.com/Luciferbobo/D4NeRF/blob/main/demo/back_ours.gif" width="200"> 
<img src="https://github.com/Luciferbobo/D4NeRF/blob/main/demo/backdepth_ours.gif" width="200"> 
  
**The Decoupled Background**
  
</div>

## Getting Started

Code will be released soon.

