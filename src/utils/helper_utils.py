import imutils
import cv2
import numpy as np

def area_rect(A, B):
    '''
    Calculated Area of Bounding Box Rectangles 
    with two opposite diagonal points.
    '''

    return(abs((B[0]-A[0])*(B[1]-A[1])))

def resize_frame(input_frame,dimensions):
    '''
    Resize input frame to model input layer dimensions
    while preserving aspect ratio of the image.

    If the resized image is smaller than the required dimensions,
    black borders are added to equalize the dimensions.

    Else if the resized image is larger than the required dimensions,
    the frame is centered and cropped to size.
    '''

    aspect_resized_frame = imutils.resize(input_frame, width = dimensions[1], inter=cv2.INTER_LINEAR)

    if(aspect_resized_frame.shape[0] > dimensions[0]):
        crop_len = (aspect_resized_frame.shape[0] - dimensions[0]) // 2
        aspect_resized_frame = aspect_resized_frame[crop_len : crop_len + aspect_resized_frame[0]][0 : dimensions[1]]
    
    elif(aspect_resized_frame.shape[0] < dimensions[0]):
        border_len = (dimensions[0] - aspect_resized_frame.shape[0]) // 2
        aspect_resized_frame = cv2.copyMakeBorder(aspect_resized_frame, border_len, border_len, 0, 0, cv2.BORDER_CONSTANT)
    
    return (aspect_resized_frame)

def roi_segment(input_frame, contour_pts):
    '''
    Extracts Region of Interest from the given frame using the user 
    specified boundary/contour points. 
    
    This is done by creating a bitmask for every frame followed by the 
    bitwise-and operator between the pixels of the bitmask and frame.
    '''
    
    stencil = np.zeros(input_frame.shape).astype(input_frame.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contour_pts, color)
    result_frame = cv2.bitwise_and(input_frame, stencil)

    return (result_frame)
