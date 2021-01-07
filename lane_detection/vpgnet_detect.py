import cv2
from wrapper_vpgnet import WrapperVPGNet
import vpgnet_utils as utils
import numpy as np

class Vpgnet():
    def __init__(self, model_path='/home/luben/csie5452-lane-detection/vpgnet_weights_1346_8_10_100_lr_1e-5/vpgnet_epoch_50.pth' ):
        self.vpgnet = WrapperVPGNet(model_path)
        # lane line detect
        self.minLineLength = 100
        self.maxLineGap = 10

        # lane line mask
        self.threshold = 0.9
        self.left_threshold = 0.999
        self.right_threshold = 0.85

        # lane line filter
        self.det_slope = 0.5

        self.color_map_mat = np.zeros((19, 3), dtype=np.uint8)
        for i in range(0, 19):
            if i == 1:
                # lane_solid_white
                self.color_map_mat[i] = (255, 255, 255)
            elif i == 3:
                # lane_double_white
                self.color_map_mat[i] = (255, 255, 255)
            elif i == 4:
                # lane_solid_yellow
                self.color_map_mat[i] = (0, 255, 255)
            elif i == 6:
                #lane_double_yellow
                self.color_map_mat[i] = (0, 255, 255)
            elif i == 18:
                # erfnet_mask
                self.color_map_mat[i] = (0, 0, 255)

    def lane_line_detect(self, image_origin, lane_solid_white, lane_double_white, lane_solid_yellow, lane_double_yellow):

        img_lane_line = image_origin.copy()
        solid_white = lane_solid_white.copy().astype(np.uint8)
        double_white = lane_double_white.copy().astype(np.uint8)
        solid_yellow = lane_solid_yellow.copy().astype(np.uint8)
        double_yellow = lane_double_yellow.copy().astype(np.uint8)

        self.draw_lanes_curve(img_lane_line, double_yellow, color=[0, 255, 255], thickness=4, point_size = 1)
        #self.draw_lanes_curve(img_lane_line, solid_white, color=[255, 255, 255], thickness=4, point_size = 1)

        lines_solid_white = cv2.HoughLinesP(solid_white, 1, np.pi / 180, 100, self.minLineLength, self.maxLineGap)
        lines_double_white = cv2.HoughLinesP(double_white, 1, np.pi / 180, 100, self.minLineLength, self.maxLineGap)
        lines_solid_yellow = cv2.HoughLinesP(solid_yellow, 1, np.pi / 180, 100, self.minLineLength, self.maxLineGap)
        lines_double_yellow = cv2.HoughLinesP(double_yellow, 1, np.pi / 180, 100, self.minLineLength, self.maxLineGap)

        #self.draw_lanes(img_lane_line, lines_double_yellow, 'lane_double_yellow', color=[0, 255, 255], thickness=4 )
        self.draw_lanes(img_lane_line, lines_solid_white, 'lane_solid_white', color=[255, 255, 255], thickness=4 )
        '''
        if isinstance(lines_solid_white,np.ndarray):
            for x1, y1, x2, y2 in lines_solid_white[0]:
                cv2.line(img_lane_line, (x1, y1), (x2, y2), (0, 0, 0), 7)
                cv2.line(img_lane_line, (x1, y1), (x2, y2), (255, 255, 255), 5)
            
        if isinstance(lines_double_yellow,np.ndarray):
            for x1, y1, x2, y2 in lines_double_yellow[0]:
                cv2.line(img_lane_line, (x1, y1), (x2, y2), (0, 0, 0), 7)
                cv2.line(img_lane_line, (x1, y1), (x2, y2), (0, 255, 255), 5)
        '''
        return img_lane_line
        
    def draw_lanes_curve(self, image_origin, lane_mask, color=[255, 0, 0], thickness=8, point_size = 1):

        lane_mask_nonzero = lane_mask.nonzero()
        nonzeroy = np.array(lane_mask_nonzero[0])
        nonzerox = np.array(lane_mask_nonzero[1])
        if len(nonzeroy) == 0 or len(nonzerox) == 0:
            return False

        lane_fit = np.polyfit(nonzeroy, nonzerox, 2)
        ploty = np.linspace(270, lane_mask.shape[0]-1, lane_mask.shape[0])
        for pty in ploty:
            ptx = lane_fit[0]*pty**2 + lane_fit[1]*pty + lane_fit[2]

            cv2.circle(image_origin, (int(ptx),int(pty)), point_size, color, thickness)
    
    def inference(self, image_origin):

        obj_mask_pred_160x120_vpgnet, vp_mask = self.vpgnet.get_lane_line_from_image_640x480(image_origin)
        lane_solid_white = utils.resize_array(obj_mask_pred_160x120_vpgnet[0, :, :], (image_origin.shape[0], image_origin.shape[1]))
        lane_double_white = utils.resize_array(obj_mask_pred_160x120_vpgnet[1, :, :], (image_origin.shape[0], image_origin.shape[1]))
        lane_solid_yellow = utils.resize_array(obj_mask_pred_160x120_vpgnet[2, :, :], (image_origin.shape[0], image_origin.shape[1]))
        lane_double_yellow = utils.resize_array(obj_mask_pred_160x120_vpgnet[3, :, :], (image_origin.shape[0], image_origin.shape[1]))


        lane_solid_white = np.where(lane_solid_white > self.right_threshold, 1, 0)
        lane_double_white = np.where(lane_double_white > self.threshold, 1, 0)
        lane_solid_yellow = np.where(lane_solid_yellow > self.threshold, 1, 0)
        lane_double_yellow = np.where(lane_double_yellow > self.left_threshold, 1, 0)


        prob_map_lane_solid_white = lane_solid_white * 1
        prob_map_lane_double_white = lane_double_white * 3
        prob_map_lane_solid_yellow = lane_solid_yellow * 4
        prob_map_lane_double_yellow = lane_double_yellow * 6

        seg_image_lane_solid_white = self.color_map_mat[prob_map_lane_solid_white]
        seg_image_lane_double_white = self.color_map_mat[prob_map_lane_double_white]
        seg_image_lane_solid_yellow = self.color_map_mat[prob_map_lane_solid_yellow]
        seg_image_lane_double_yellow = self.color_map_mat[prob_map_lane_double_yellow]

        gamma = 0

        img_lane_mask = cv2.addWeighted(seg_image_lane_double_yellow, 1, seg_image_lane_solid_white, 1, gamma)
        #img_lane_mask = cv2.addWeighted(seg_image_lane_solid_yellow, 1, img_lane_mask, 1, gamma)
        #img_lane_mask = cv2.addWeighted(seg_image_lane_double_white, 1, img_lane_mask, 1, gamma)
        img_lane_mask = cv2.blur(img_lane_mask, (9, 9))

        img_lane_line = self.lane_line_detect(image_origin, lane_solid_white, lane_double_white, lane_solid_yellow, lane_double_yellow)
        img_lane_all = cv2.addWeighted(img_lane_line, 0.8, img_lane_mask, 0.7, gamma)


        return img_lane_all

    def draw_lanes(self, img, lines, line_cls, color=[255, 0, 0], thickness=8):
        filter_lines = []
        if lines is None:
            return False
        for line in lines:
            for x1, y1, x2, y2 in line:
                k = (y2 - y1) / (x2 - x1)
                if line_cls == 'lane_double_yellow':
                    if k < -self.det_slope:
                        filter_lines.append(line)
                elif line_cls == 'lane_solid_white':
                    if k > self.det_slope:
                        filter_lines.append(line)

        if (len(filter_lines) <= 0):
            return img

        self.clean_lines(filter_lines, 0.1)#弹出左侧不满足斜率要求的直线
        #self.clean_lines(right_lines, 0.1)#弹出右侧不满足斜率要求的直线
        points = [(x1, y1) for line in filter_lines for x1,y1,x2,y2 in line]#提取左侧直线族中的所有的第一个点
        points = points + [(x2, y2) for line in filter_lines for x1,y1,x2,y2 in line]#提取左侧直线族中的所有的第二个点
        #right_points = [(x1, y1) for line in right_lines for x1,y1,x2,y2 in line]#提取右侧直线族中的所有的第一个点
        #right_points = right_points + [(x2, y2) for line in right_lines for x1,y1,x2,y2 in line]#提取右侧侧直线族中的所有的第二个点

        vtx = self.calc_lane_vertices(points, 270, img.shape[0])#拟合点集，生成直线表达式，并计算左侧直线在图像中的两个端点的坐标
        #right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])#拟合点集，生成直线表达式，并计算右侧直线在图像中的两个端点的坐标

        cv2.line(img, vtx[0], vtx[1], color, thickness)#画出直线
        #cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)#画出直线

    #将不满足斜率要求的直线弹出
    def clean_lines(self, lines, threshold):
        slope=[]
        for line in lines:
            for x1,y1,x2,y2 in line:
                k=(y2-y1)/(x2-x1)
                slope.append(k)
        #slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
        while len(lines) > 0:
            mean = np.mean(slope)#计算斜率的平均值，因为后面会将直线和斜率值弹出
            diff = [abs(s - mean) for s in slope]#计算每条直线斜率与平均值的差值
            idx = np.argmax(diff)#计算差值的最大值的下标
            if diff[idx] > threshold:#将差值大于阈值的直线弹出
                slope.pop(idx)#弹出斜率
                lines.pop(idx)#弹出直线
            else:
                break

    #拟合点集，生成直线表达式，并计算直线在图像中的两个端点的坐标
    def calc_lane_vertices(self, point_list, ymin, ymax):
        x = [p[0] for p in point_list]#提取x
        y = [p[1] for p in point_list]#提取y
        fit = np.polyfit(y, x, 1)#用一次多项式x=a*y+b拟合这些点，fit是(a,b)
        fit_fn = np.poly1d(fit)#生成多项式对象a*y+b

        xmin = int(fit_fn(ymin))#计算这条直线在图像中最左侧的横坐标
        xmax = int(fit_fn(ymax))#计算这条直线在图像中最右侧的横坐标

        return [(xmin, ymin), (xmax, ymax)]


if __name__ == '__main__':
    vpgnet = Vpgnet()

    video_path = '/home/luben/car-data/demo4.mp4'
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))

            result_frame = vpgnet.inference(frame)
            cv2.imshow('frame',result_frame)
        if not ret:
            print('can\'t receive frame')
            break
    
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
