import time

import cv2
import numpy as np

# read first 2 frames of a video and convert to grayscale
cap = cv2.VideoCapture('opticalflowtest.mp4')
ret, prev = cap.read()
h, w = prev.shape[:2]
crop_box = (w*0.3, h*0.6, w*0.4, h*0.4)  # xywh (take middle 40% of x axis (w) and bottom 30% of y axis (h))

# crop out the floor part only
prev = prev[int(crop_box[1]):int(crop_box[1]+crop_box[3]), int(crop_box[0]):int(crop_box[0]+crop_box[2]), :]
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    # create line endpoints
    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # create image and draw
    cv2.polylines(img, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    return img

while True:
    start = time.time()
    ret, new = cap.read()
    new = new[int(crop_box[1]):int(crop_box[1]+crop_box[3]), int(crop_box[0]):int(crop_box[0]+crop_box[2]), :]
    new_gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    # prev, next, flow (pointer in C, use None in Python), pyrscale, levels, winsize, iterations, poly_n, poly_sigma, flags
    flow = cv2.calcOpticalFlowFarneback(prev_gray, new_gray, None, 0.5, 3, 5, 15, 5, 1.2, 0)  # if tracking not sensitive enough, increase iterations
    # flow is of shape (h, w, 2), where flow[:,:,0] is the x axis movement and flow[:,:,1] is the y axis movement
    prev_gray = new_gray
    end = time.time()
    fps = 1 / (end - start)
    drawn = draw_flow(new, flow)
    cv2.putText(drawn, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.putText(drawn, f"FPS: {fps:.2f}", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('OpticalFlow', drawn)
    flow_y_mean, flow_x_mean = np.mean(flow, axis=(0, 1))  # take the mean of x and y axis movement
    print(flow_y_mean, flow_x_mean)
    cv2.waitKey(1)
