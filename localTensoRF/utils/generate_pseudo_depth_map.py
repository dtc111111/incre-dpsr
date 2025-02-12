import cv2

color_image = cv2.imread('/data/localrf-main-latest/localTensoRF/utils/002408.jpg')

gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

colored_depth_map = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

cv2.imwrite('/data/localrf-main-latest/localTensoRF/utils/colored_depth_map.png', colored_depth_map)
