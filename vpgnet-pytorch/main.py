import cv2
from wrapper_vpgnet import WrapperVPGNet
import utils
model_path = '/home/luben/CSIE5452_FinalProj/vpgnet-pytorch/torch/vpgnet_weights/vpgnet_01.pth'
vpgnet = WrapperVPGNet(model_path)
image_origin = cv2.imread('lane_scene_01.jpg', cv2.IMREAD_COLOR)
copy_image_origin_vpgnet = cv2.resize(image_origin.copy(), (640, 480))
obj_mask_pred_160x120_vpgnet = vpgnet.get_lane_line_from_image_640x480(copy_image_origin_vpgnet)

# obj_mask_pred_1920x1080_vpgnet = np.resize(obj_mask_pred_160x120_vpgnet, (4, 1080, 1920))
# vpgnet车道线开始
#lane_solid_white = utils.resize_array(obj_mask_pred_160x120_vpgnet[0, :, :], (1080, 1920))
#lane_solid_yellow = utils.resize_array(obj_mask_pred_160x120_vpgnet[1, :, :], (1080, 1920))
#lane_double_yellow = utils.resize_array(obj_mask_pred_160x120_vpgnet[2, :, :], (1080, 1920))
lane_mask = utils.resize_array(obj_mask_pred_160x120_vpgnet[0, :, :], (1080, 1920))
print(obj_mask_pred_160x120_vpgnet.shape)
#cv2.imshow('lane mask', obj_mask_pred_160x120_vpgnet[0,:,:])
cv2.imshow('lane mask', lane_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
