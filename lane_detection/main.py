import cv2
from yolo_detect import Yolov3
from vpgnet_detect import Vpgnet
from utils.plots import plot_one_box
import numpy as np
import random

if __name__ == '__main__':

    vpgnet_model_path = '/home/luben/csie5452-lane-detection/vpgnet_weights_123456_8_10_150_lr_1e-5/vpgnet_epoch_45.pth'
    #vpgnet_model_path = '/home/luben/csie5452-lane-detection/vpgnet_weights_1346_8_10_100_lr_1e-5/vpgnet_epoch_50.pth'
    vpgnet = Vpgnet(vpgnet_model_path)
    yolo = Yolov3()

    obj_tracked = 'person'
    video_path = '/home/luben/car-data/demo4.mp4'
    cap = cv2.VideoCapture(video_path)

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
    tracker_type = tracker_types[1]
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
    left_curve = 1
    init = 1
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            # Start timer
            timer = cv2.getTickCount()

            frame = cv2.resize(frame, (640, 480))
            result_frame, double_yellow_fit = vpgnet.inference(frame, left_curve=left_curve)
            img, detect_bboxes, confidence, label_name = yolo.detect(frame.copy(), save_img=False)

            # init frame
            if init:
                for idx, detect_bbox in enumerate(detect_bboxes):

                    if label_name[idx] == obj_tracked:
                        xyxy = detect_bbox
                        xywh = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])]
                        ok = tracker.init(frame, xywh)
                        init = 0
            else:
                ok, track_bbox = tracker.update(frame)
                if ok:
                    detect_center = (0,0)
                    for idx, detect_bbox in enumerate(detect_bboxes):
                        if label_name[idx] == obj_tracked:
                            detect_center = (int((detect_bbox[0]+detect_bbox[2])/2), int((detect_bbox[1]+detect_bbox[3])/2))

                    cv2.circle(result_frame, detect_center, 3, (255, 0, 0), 3)

                    # Tracking success
                    p1 = (int(track_bbox[0]), int(track_bbox[1]))
                    p2 = (int(track_bbox[0] + track_bbox[2]), int(track_bbox[1] + track_bbox[3]))
                    cv2.rectangle(result_frame, p1, p2, (255, 0, 0), 2, 1)
                    track_center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
                    cv2.circle(result_frame, track_center, 3, (0, 0, 255), 3)

                    pty = track_center[1]
                    print(pty)
                    if isinstance(double_yellow_fit,np.ndarray):
                        ptx = 0
                        if left_curve == 1:
                            ptx = double_yellow_fit[0] * pty ** 2 + double_yellow_fit[1] *pty + double_yellow_fit[2]
                        else:
                            ptx = double_yellow_fit[0] * pty + double_yellow_fit[1]
                        diff = ptx - track_center[0]
                        if diff > 0 :
                            cv2.putText(result_frame, str(diff), (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 255), 2)
                            cv2.putText(result_frame, 'Warning', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
                        else:
                            cv2.putText(result_frame, str(diff), (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255, 0, 0), 2)

                else:
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # plot all bboxes
            '''
            for idx,bbox in enumerate(bboxes):
                color = [0, 255, 0]
                center = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))
                cv2.circle(frame, center, 3, (0, 0, 255), 3)
                label_text = label_name[idx] + '  '+ str(confidence[idx])
                plot_one_box(bbox, frame, label=label_text, color=color , line_thickness=2)
            '''

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
            # Display FPS on frame
            cv2.putText(result_frame, "FPS : " + str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)



            cv2.imshow('frame',result_frame)
        if not ret:
            print('can\'t receive frame')
            break
    
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
