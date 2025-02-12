# incre-dpsr
We propose an incremental joint learning framework, which can achieve accurate depth, pose estimation, and large-scale dense scene reconstruction. For depth estimation, a vision transformer-based network is adopted as the backbone to enhance performance in scale information estimation. For pose estimation, a feature-metric bundle adjustment (FBA) method is designed for accurate and robust camera tracking in large-scale scenes, and eliminate the pose drift. In terms of implicit scene representation, we propose an incremental scene representation method to construct the entire large-scale scene as multiple local radiance fields to enhance the scalability of 3D scene representation. In local radiance fields, we propose a tri-plane based scene representation method to further improve the accuracy and efficiency of scene reconstruction. We conduct extensive experiments on various datasets, including our own collected data, to demonstrate the effectiveness and accuracy of our method in depth estimation, pose estimation, and large-scale scene reconstruction.
## Setup
```
git clone https://github.com/dtc111111/incre-dpsr.git
cd incre-dpsr
conda env create -f environment.yaml
conda activate incre-dpsr
```
