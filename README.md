# Incremental Joint Learning of Depth, Pose and Implicit Scene Representation on Monocular Camera in Large-scale Scenes \\ [TASE 2025] 
We propose an incremental joint learning framework, which can achieve accurate depth, pose estimation, and large-scale dense scene reconstruction. For depth estimation, a vision transformer-based network is adopted as the backbone to enhance performance in scale information estimation. For pose estimation, a feature-metric bundle adjustment (FBA) method is designed for accurate and robust camera tracking in large-scale scenes, and eliminate the pose drift. In terms of implicit scene representation, we propose an incremental scene representation method to construct the entire large-scale scene as multiple local radiance fields to enhance the scalability of 3D scene representation. In local radiance fields, we propose a tri-plane based scene representation method to further improve the accuracy and efficiency of scene reconstruction. We conduct extensive experiments on various datasets, including our own collected data, to demonstrate the effectiveness and accuracy of our method in depth estimation, pose estimation, and large-scale scene reconstruction.

## Setup
```
git clone https://github.com/dtc111111/incre-dpsr.git
cd incre-dpsr
conda env create -f environment.yaml
conda activate incre-dpsr
```

## Datasets
1. [Tanks and Temples](https://www.robots.ox.ac.uk/~wenjing/Tanks.zip)

We use eight scenes from Tanks and Temples in our paper: Ballroom, Barn, Church, Family, Francis, Horse, Ignatius and Museum.

2. [Static Hikes](https://drive.google.com/file/d/1DngTRNuZZXpho8-2cjpToa3lGWzxgwqL/view?usp=drive_link)

Our paper also validates the effectiveness on the Static Hikes dataset.

## Preprocessing
For each scene, we use [DPT](https://github.com/isl-org/DPT) and [RAFT](https://github.com/princeton-vl/RAFT) for monocular depth prior and flow.
```
bash scripts/download_weights.sh ## Get pretrained weights.
python scripts/run_flow.py --data_dir ${SCENE_DIR} ## Run flow estimation (assuming sorted image files in `${SCENE_DIR}/images`).
python DPT/run_monodepth.py --input_path ${SCENE_DIR}/images --output_path ${SCENE_DIR}/depth --model_type dpt_large ## Run depth estimation.
```

## Pose Estimation
We use Feature Bundle Adjustment(FBA) method to optimize pose estimation. 

First a [Unet](~/localTensoRF/models/unet.py) is used to extract the multi-layer feature maps of RGB images.
With the multi-layer feature information and confidence matrix, we then use FBA method to calculate and minimize the reprojection error

```

```


## Training
```
python localTensoRF/train.py --datadir ${SCENE_DIR} --logdir ${LOG_DIR} --fov ${FOV}
```
`${LOG_DIR}` is for the test views and smoothed trajectories to be stored.




