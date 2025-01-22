import os
import cv2
import numpy as np

input_folder = '/dataset/Tanks/Dataset_ours_2/depth'  # 输入文件夹路径
output_folder = '/dataset/Tanks/Dataset_ours_2/dpt'  # 输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 批量处理深度图像
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith('.png'):
        depth_png_file = os.path.join(input_folder, filename)
        depth_npz_file = os.path.join(output_folder, 'depth_' + filename[:-4] + '.npz')  # 使用与输入文件相同的名称，但后缀改为.npz

        # 加载.png格式深度图像，并保存为.npz格式
        depth_png = cv2.imread(depth_png_file, cv2.IMREAD_ANYDEPTH)
        depth_array = depth_png.astype(np.float32) / 1000.0  # 深度值除以1000进行单位转换

        # 将深度图像数组保存为.npz格式文件
        np.savez_compressed(depth_npz_file, depth=depth_array)
