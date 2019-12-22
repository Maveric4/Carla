
## imports
import cv2
import time
import numpy as np


def region_of_interest(img, vertices):  # not
    mask = np.zeros_like(img)
    ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def detect_lanes(image):
    imshape = image.shape
    control_decision = [0, 0, 0]
    lower_left = [0, imshape[0] - imshape[0]//9] # do dopracowania
    lower_right = [imshape[1], imshape[0] - imshape[0]//9]
    top_left = [imshape[1] // 2 - imshape[1] // 5, imshape[0] // 5 * 3]  # - imshape[0] // 5]
    top_right = [imshape[1] // 2 + imshape[1] // 5, imshape[0] // 5 * 3]  # - imshape[0] // 5]

    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)[:, :, 2]
    # im_bin = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY_INV)

    height, width = im_gray.shape

    im_res = np.zeros((height, width, 1), np.uint8)

    im_gray_ = cv2.blur(im_gray, (5, 5))

    im_line = cv2.Canny(im_gray_, 60, 120)
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
    im_line = region_of_interest(im_line, vertices)

    # im_line = cv2.Sobel()

    # im_hough = cv2.imread('data/foto3.jpg')
    # im_hough = cv2.resize(im_hough, (0,0), fx=0.5, fy=0.5)

    # lines = cv2.HoughLines(im_line,1,np.pi/180,200)

    lines = cv2.HoughLines(im_line, 1, np.pi / 90, 50)
    # minLineLength = 100
    # maxLineGap = 10
    # lines = cv2.HoughLinesP(im_line, 1, np.pi/180, 100, minLineLength, maxLineGap)

    cnt = 0

    rhos = []
    thetas = []
    was_here = False
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if theta > 1.3 and theta < 2:
                    continue
                for r, t in zip(rhos, thetas):
                    if abs(r - rho) < 40:
                        if abs(t - theta) < 0.3:
                            was_here = True

                if was_here is True:
                    was_here = False
                    continue

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                a_coeff = -b/a
                x1 = int(x0 + 2000 * (-b))
                y1 = int(y0 + 2000 * (a))
                x2 = int(x0 - 2000 * (-b))
                y2 = int(y0 - 2000 * (a))
                if theta <= 1.3:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    left_lines.append([a_coeff, rho/a])
                if theta >= 2:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    right_lines.append([a_coeff, rho/a])
                rhos.append(rho)
                thetas.append(theta)

    l1 = [item[0] for item in left_lines]
    l2 = [item[1] for item in left_lines]
    r1 = [item[0] for item in right_lines]
    r2 = [item[1] for item in right_lines]
    cv2.circle(image, tuple(top_left), 3, (255, 255, 0))
    cv2.circle(image, tuple(top_right), 3, (255, 255, 0))
    cv2.circle(image, tuple(lower_left), 3, (255, 255, 0))
    cv2.circle(image, tuple(lower_right), 3, (255, 255, 0))
    common_point = 0
    try:
        [a1, b1] = [sum(l1)/len(l1), sum(l2)/len(l2)]
        [a2, b2] = [sum(r1)/len(r1), sum(r2)/len(r2)]
        common_point = (int(a1*((b2-b1)/(a1-a2))+b1), int((b2-b1)/(a1-a2)))
        # print(common_point[0] - image.shape[1]//2)
        cv2.circle(image, common_point, 10, (255, 255, 255))
        control_decision = [(common_point[0] - image.shape[1]//2) / image.shape[1], 0.5, 0]
        # self._outq_control.put([(common_point[0] - image.shape[1]//2) * 50, 50000, 64000])
    except ZeroDivisionError:
        print("error")
        pass

    # cv2.imshow('Original', image)
    # cv2.imshow('Lines', im_line)
    # cv2.waitKey(1)

    return control_decision


