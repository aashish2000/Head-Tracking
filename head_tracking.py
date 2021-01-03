'''
Contains the Head Detection and Tracking Functionality of the Application.

Uses a pre-trained SSD for Head Detection and performs Head Tracking using
the Centroid Tracking Algorithm.
'''

# Perform necessary imports
import os
import datetime
import cv2
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

from src.models.keras_ssd512 import ssd_512
from src.keras_loss_function.keras_ssd_loss import SSDLoss
from src.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from src.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from src.keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from src.keras_layers.keras_layer_L2Normalization import L2Normalization
from src.head_tracker.centroidtracker import CentroidTracker
from src.utils.helper_utils import resize_frame, roi_segment, area_rect

def video_inference(video_path):

    # This model uses a 512x512 input size, let's set our image size to that resolution size
    img_height = 512
    img_width = 512

    # Set the model's inference mode
    model_mode = 'inference_fast'

    # Finally, we set the model's confidence threshold value, i.e. the model will only output predictions 
    # above or equal to this value. Set a value in range [0, 1], where a higher value decreases the number 
    # of detections produced by the model and increases its speed during inference. 
    # 
    # We choose to set a low value (1%) for the model confidence threshold and then use a higher value at a later stage. 
    conf_thresh = 0.01

    # Make sure the path correctly points to the model's .h5 file
    weights_path = './data/ssd512-hollywood-trainval-bs_16-lr_1e-05-scale_pascal-epoch-187-py3.6.h5'

    # Create an SSDLoss object in order to pass that to the model loader
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    # Clear previous models from memory.
    K.clear_session() 

    # Configure the decode detections layer based on the model mode
    if model_mode == 'inference':
        decode_layer = DecodeDetections(img_height=img_height,
                                        img_width=img_width,
                                        confidence_thresh=conf_thresh,
                                        iou_threshold=0.2,
                                        top_k=200,
                                        nms_max_output_size=400)
    if model_mode == 'inference_fast':
        decode_layer = DecodeDetectionsFast(img_height=img_height,
                                            img_width=img_width,
                                            confidence_thresh=conf_thresh,
                                            iou_threshold=0.2,
                                            top_k=200,
                                            nms_max_output_size=400)
        
    # Finally load the model
    model = load_model(weights_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                    'L2Normalization': L2Normalization,
                                                    'DecodeDetections': decode_layer,
                                                    'compute_loss': ssd_loss.compute_loss})


    start = datetime.datetime.now()

    # Set final confidence threshold for boundign box prediction
    confidence_threshold = 0.45

    # Initalize Centroid Tracker object for Head Tracking
    ct = CentroidTracker()

    (H, W) = (img_height, img_width)
    classes = ['background', 'head']
    save_path = 'output-{}.webm'.format(datetime.datetime.timestamp(start))

    cap = cv2.VideoCapture(video_path)
    save = cv2.VideoWriter("./video_files/results/"+save_path, cv2.VideoWriter_fourcc('V','P','8','0'), 30, (512,512))

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    frame = None
    if cap.isOpened():
        hasFrame, frame = cap.read()
    else:
        hasFrame = False

    # Now we feed the example images into the network to obtain the predictions.
    head_cnt = 0
    area_rect_frame = []
    while hasFrame:
        # Resize input frame to 512x512
        frame = resize_frame(frame, (H,W))
        
        # Perform ROI Segmentation from input frame
        contours = [np.array([[0, 210], [180, 200], [230,200], [512, 200], [512, 448],[0,448]])]
        frame = roi_segment(frame, contours)
        
        # Invert image channels from BGR to RGB anf run model inference
        orig_images = [frame[...,::-1].astype(np.float32)]
        y_pred = model.predict(np.array(orig_images))

        # Perform confidence thresholding.
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

        rects = []
        for box in y_pred_thresh[0]:
            # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
            xmin = box[2] * np.array(orig_images[0]).shape[1] / img_width
            ymin = box[3] * np.array(orig_images[0]).shape[0] / img_height
            xmax = box[4] * np.array(orig_images[0]).shape[1] / img_width
            ymax = box[5] * np.array(orig_images[0]).shape[0] / img_height

            # Remove false predictions based on Bounding box size
            if(area_rect((int(xmin),int(ymin)),(int(xmax),int(ymax))) > 3000):
                rects.append((int(xmin),int(ymin),int(xmax),int(ymax)))
                area_rect_frame.append(area_rect((int(xmin),int(ymin)),(int(xmax),int(ymax))))
                frame = cv2.rectangle(frame,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255, 0, 0),4)
            
        objects = ct.update(rects)

        # Loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # Draw both the ID of the object and the centroid of the object on the output frame
            head_cnt = max(objectID+1, head_cnt)
            text = "Head: {}".format(objectID+1)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        save.write(frame)
        hasFrame, frame = cap.read()

        # Update the Head Counter .
        frame = cv2.putText(frame,"Heads: "+str(head_cnt), (img_width-150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 200), 2)

    cap.release()
    save.release()

    print("Time Elapsed:",datetime.datetime.now()-start)
    return (save_path,head_cnt)
                
