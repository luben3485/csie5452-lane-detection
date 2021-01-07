import cv2
from yolo_detect import Yolov3
from vpgnet_detect import Vpgnet

if __name__ == '__main__':


    vpgnet_model_path = '/home/luben/csie5452-lane-detection/vpgnet_weights_1346_8_10_100_lr_1e-5/vpgnet_epoch_50.pth'
    vpgnet = Vpgnet(vpgnet_model_path)
    yolo = Yolov3()

    video_path = '/home/luben/car-data/demo4.mp4'
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            img, bbox , label_list= yolo.detect(frame, save_img=False) 
            result_frame = vpgnet.inference(frame)
            cv2.imshow('frame',result_frame)
        if not ret:
            print('can\'t receive frame')
            break
    
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
