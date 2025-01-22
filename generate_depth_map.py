import cv2
import torch
import os
from matplotlib.pyplot import get_cmap
import numpy
#from localTensoRF.utils import visualize_depth
if __name__ == "__main__":
    #depth_map = cv2.imread("/data/localrf-main-latest/localTensoRF/utils/002401.png", cv2.IMREAD_GRAYSCALE)
    #depth_map = depth_map.reshape(540, 960)
    #depth_map_vis, _ = visualize_depth(depth_map, [0, 5])
    #depth_map_vis = torch.permute(depth_map_vis * 255, [1, 2, 0]).byte()
    #depth_map_vis = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    #cv2.imwrite("/data/localrf-main-latest/localTensoRF/utils/depth_map.jpg", depth_map_vis)
    image_paths = sorted(os.listdir("/data/localrf-main-latest/localTensoRF/utils/dpt"))
    for image_path in image_paths:
        image_fbase = os.path.splitext(image_path)[0]
        depth_map = cv2.imread(os.path.join("/data/localrf-main-latest/localTensoRF/utils/dpt", image_path), cv2.IMREAD_GRAYSCALE)
        x=depth_map
        #x = numpy.nan_to_num(depth_map)
        min = numpy.min(x[x > 0])  # get minimum positive depth (ignore background)
        max = numpy.max(x)
        x = (x - min) / (max - min + 1e-8)  # normalize to 0~1
    
        x = (255 * numpy.clip(x, 0.17, 1)).astype(numpy.uint8)
        #depth_map_vis = cv2.applyColorMap(x, cv2.COLORMAP_JET)
        depth_map_vis = (255*get_cmap('jet')(x)).astype('uint8')

        #folder_name = "/data/localrf-main-latest/localTensoRF/utils/depth_maps"
        #file_name = 
        
        cv2.imwrite("{}.jpg".format(os.path.join("/data/localrf-main-latest/localTensoRF/utils/depth_maps", image_fbase)), depth_map_vis)
