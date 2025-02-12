import cv2
import torch
import os
from matplotlib.pyplot import get_cmap
from utils import visualize_depth
if __name__ == "__main__":
    depth_map = cv2.imread("/data/localrf-main-latest/localTensoRF/utils/1.png", cv2.IMREAD_GRAYSCALE)
    depth_map = depth_map.reshape(540, 960)
    depth_map_vis, _ = visualize_depth(depth_map, [0, 5])
    depth_map_vis = torch.permute(depth_map_vis * 255, [1, 2, 0]).byte()
    depth_map_vis = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    cv2.imwrite("/data/localrf-main-latest/localTensoRF/utils/depth_map.jpg", depth_map_vis)
    '''
    image_paths = sorted(os.listdir("/data/localrf-main-latest/localTensoRF/utils/nope_depth_out"))
    for image_path in image_paths:
        image_fbase = os.path.splitext(image_path)[0]
        depth_map = cv2.imread(os.path.join("/data/localrf-main-latest/localTensoRF/utils/nope_depth_out", image_path), cv2.IMREAD_GRAYSCALE)
        #depth_map_vis = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        depth_map_vis = (255.0 * get_cmap('jet')(depth_map)).astype('uint8')
        #folder_name = "/data/localrf-main-latest/localTensoRF/utils/depth_maps"
        #file_name = 
        
        cv2.imwrite("{}.jpg".format(os.path.join("/data/localrf-main-latest/localTensoRF/utils/nope_depth_maps", image_fbase)), depth_map_vis)
    '''