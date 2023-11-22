# TMM-MMH
Code for Privileged Modality Learning via Multimodal Hallucination (TMM)

# Face anti-spoofing task
## DATASET
- DOWNLODAD LINK https://sites.google.com/view/face-anti-spoofing-challenge/welcome/challengecvpr2019
  - Require sign a agreement
## Run
  - parameter 
    - **data_root**: path to your dataset, such as '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF'
    - **modal**: the modality used, such as rgb,depth,ir
    - **backbone**: name of backbone, it is only the name determined by your backbone,  and can not to select backbone
  - single baseline
    ```bash
    cd src
    bash baseline_depth.sh
    bash baseline_ir.sh
    bash baseline_rgb.sh
    ```
    
  - multimodal teacher model  
    ```bash
    cd src
    bash baseline_multi.sh
    bash cefa_baseline_multi.sh
    ``` 
  - multimodal model hallucination
    - RAD
      ```bash
      cd src
      bash surf_patch_spp.sh
      bash cefa_patch_kd_spp.sh
      ```
    - DAD
      ```bash
      cd src
      bash surf_patch_feature.sh
      bash cefa_patch_kd_feature.sh
      ```

# Action Recognition
## Dataset
- [NW-UCLA](https://ieeexplore.ieee.org/document/6909735/references#references)
- [NTUD60](https://ieeexplore.ieee.org/document/7780484)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

## RUN
- multimodal teacher model
  - see code in https://github.com/shicaiwei123/pytorch-i3d
- multimodal model hallucination
  - UCF101
    - ```bash
      cd ucf101/src
      bash ucf101_patch_spp.sh
        ```
  - HMDB51
    - ```bash
      cd hmdb51/src
      bash hmdb51_patch_spp.sh
        ```
  - NWUCLA
    - ```bash
      cd nw_ucla_cv
      bash ucla_patch_spp.sh
      ```
  - NTUD60
    - ```bash
      cd ntud60_cs
      bash ntud60_patch_spp.sh
      ```