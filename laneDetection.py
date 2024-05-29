import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

def birdsEyeView(image):
    """
    Schimba perspectiva imaginii in birdsEyeView
    """

    top_left = [570, 458]
    top_right = [720, 458]
    bottom_left = [208, 665]
    bottom_right = [1150, 665]

    width, height = 1280, 220

    pts1 = np.float32([top_left, top_right, bottom_left, bottom_right])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    perspective_correction = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_correction_inv = cv2.getPerspectiveTransform(pts2, pts1)

    result = cv2.warpPerspective(image, perspective_correction, (width, height), flags=cv2.INTER_LANCZOS4)

    return result, perspective_correction_inv

def edgeDetection(image):
    """
    Pentru edge detection am folosit operatorul Scharr, care detecteaza derivate, gasind astfel diferentele de culoare din imagine
    """

    b, g, r = cv2.split(image)

    channel = g
    edge = cv2.Scharr(channel, cv2.CV_64F, 1, 0)
    edge = np.absolute(edge) #folosim modulul pentru ca Scharr returneaza atat valori pozitive cat si negative, dar noi vrem sa stim doar daca avem un edge
    edge = np.uint8(255 * edge / np.max(edge)) #convertim valorile in int si le facem sa fie intr-un interval 0-255, cum este necesar pentru o imagine cu un singur canal

    return edge

def applyDynamicThresholds(image):
    """
    Am aplicat un threshold mai mare in partea de jos, unde imaginea e mai clara si un threshold mai mic in partea de sus, unde pixelii sunt mai 
    distorsionati
    """
    binary = np.zeros_like(image)
    #binary[image >= 50] = 255
    threshold_up = 15
    threshold_down = 60
    threshold_delta = threshold_down-threshold_up
    for y in range(220):
        threshold_line = threshold_up + threshold_delta * y / 220
        binary[y, image[y, :] >= threshold_line] = 255
    return binary

def applyHSLThreshold(image):
    """
    Am folosit un threshold de 140 pentru channel-ul L din imaginea originala in format HSL
    """
    imghsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l_channel = imghsl[:,:,1]
    _, thresholded_l_channel = cv2.threshold(l_channel, 140, 255, cv2.THRESH_BINARY)
    return thresholded_l_channel

def generateHistogram(image):
    partial_img = image[image.shape[0] // 2:, :]
    hist = np.sum(partial_img, axis=0)
    size = len(hist)
    max_index_left = np.argmax(hist[0:size//2])
    max_index_right = np.argmax(hist[size//2:]) + size//2
    return hist

def calculate_steering_angle(hist):
    """ Am calculat steering angle-ul folosindu-ne de valorile maxime din histograma, care ne indica unde ar fi benzile"""
    midpoint = len(hist) // 2
    left_peak = np.argmax(hist[:midpoint])
    right_peak = np.argmax(hist[midpoint:]) + midpoint
    offset = (left_peak + right_peak) / 2 - midpoint
    max_offset = midpoint / 2
    steering_angle = np.arctan(offset / max_offset) * 45 / np.pi
    return steering_angle

def draw_lane_lines_on_original(image, result, hist, perspective_correction_inv):
    size = len(hist)
    max_index_left = np.argmax(hist[:size//2])
    max_index_right = np.argmax(hist[size//2:]) + size//2
    
    y_bottom = result.shape[0] - 1
    left_line_bottom = [max_index_left, y_bottom]
    right_line_bottom = [max_index_right, y_bottom]
    left_line_top = [max_index_left, 0]
    right_line_top = [max_index_right, 0]

    pts_bird_eye = np.float32([left_line_bottom, left_line_top, right_line_bottom, right_line_top])
    pts_original = cv2.perspectiveTransform(pts_bird_eye[None, :, :], perspective_correction_inv)[0]

    pts_original = pts_original.astype(int)

    roadImage_with_lines = image.copy()
    cv2.line(roadImage_with_lines, tuple(pts_original[0]), tuple(pts_original[1]), (0, 255, 0), 5)
    cv2.line(roadImage_with_lines, tuple(pts_original[2]), tuple(pts_original[3]), (0, 255, 0), 5)

    steering_angle = calculate_steering_angle(hist)
    
    cv2.putText(roadImage_with_lines, f'Steering Angle: {steering_angle:.2f} degrees', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return roadImage_with_lines

cap = cv2.VideoCapture('video_test.mp4')

ret, frame = cap.read()

height, width, _ = frame.shape

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

while True:
    ret, frame = cap.read()

    if not ret:
        break
    birds, perspective_correction_inv = birdsEyeView(frame)
    edge = edgeDetection(birds)
    thresh1 = applyDynamicThresholds(edge)
    thresh2 = applyHSLThreshold(birds)
    thresholdsCombined = cv2.bitwise_or(thresh1, thresh2)

    hist = generateHistogram(thresholdsCombined)
    frame_with_lines = draw_lane_lines_on_original(frame, birds, hist, perspective_correction_inv)
    out.write(frame_with_lines)
    
cap.release()
out.release()

print("Video processed successfully!")