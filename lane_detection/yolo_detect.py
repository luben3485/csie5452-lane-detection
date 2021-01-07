import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np

class Yolov3():

    def __init__(self,device=''):

        weights = 'yolov3-tiny.pt'

        # Initialize
        self.device = device
        
        set_logging()
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        if half:
            self.model.half()  # to FP16


    def detect(self, _img, save_img=False):

        model = self.model


        save_txt = False
        save_img = True
        imgsz = 640
        conf_thres = 0.25
        iou_thres = 0.45
        classes=None
        agnostic_nms=False
        view_img = False

        # Initialize
        set_logging()
        device =''
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        bbox_list = []
        label_list = []


        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size


        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None


        dataset = LoadImages(_img, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)


            # Process detections
            for i, det in enumerate(pred):  # detections per image

                s, im0 = '', im0s



                #save_path = str(save_dir / p.name)
                #txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                

                


                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    


                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        #if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)


                        for item in xyxy:
                            #print(item)
                            bbox_list.append(item.cpu().numpy())

                        label_list.append(label)
                        bbox_array = np.array(bbox_list)
                        bbox_array = bbox_array.reshape(-1,4)


                # Print time (inference + NMS)
                #print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results



                if view_img:
                    cv2.imshow("QQ", im0)
                    if cv2.waitKey(0) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)

        return im0, bbox_array, label_list

if __name__ == '__main__':
    _img = cv2.imread('/home/chaco/Desktop/car2/yolov3/data/images/scene002.jpg')

    #print(_img)

    with torch.no_grad():
        model = Yolov3()
        img, bbox , label_list= model.detect(_img, save_img=False)

        print("===============================================================")
        print(len(bbox))

        for idx, item in enumerate(bbox):
            print(item)
            print(label_list[idx])
        #print(len(bbox))

        cv2.imshow("QQ", img)
        if cv2.waitKey(0) == ord('q'):  # q to quit
            raise StopIteration
