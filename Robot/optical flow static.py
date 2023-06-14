import cv2
import numpy as np


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

prev = cv2.imread('data/imgs/image_0000.png')
h, w = prev.shape[:2]
# crop out the floor part only
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# prev, next, flow (pointer in C, use None in Python), pyrscale, levels, winsize, iterations, poly_n, poly_sigma, flags
flow = cv2.calcOpticalFlowFarneback(prev_gray, prev_gray, None, 0.5, 3, 5, 15, 5, 1.2, 0)
drawn = draw_flow(prev, flow)
cv2.imwrite('flow.png', drawn)
flow_y_mean, flow_x_mean = np.mean(flow, axis=(0, 1))  # take the mean of x and y axis movement
print(flow_y_mean, flow_x_mean)
cv2.waitKey(1)
