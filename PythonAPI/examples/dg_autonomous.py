
## imports
import cv2
import time
import numpy as np


def region_of_interest2(img, vertices):  # not
    mask = np.zeros_like(img)
    ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def detect_lanes(image):
    imshape = image.shape
    control_decision = [0, 0.7, 0]
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
    im_line = region_of_interest2(im_line, vertices)

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
        control_decision = [(common_point[0] - image.shape[1]//2) * 10/ image.shape[1], 1, 0]
        # self._outq_control.put([(common_point[0] - image.shape[1]//2) * 50, 50000, 64000])
    except ZeroDivisionError:
        pass

    # cv2.imshow('Original', image)
    # cv2.imshow('Lines', im_line)
    # cv2.waitKey(1)

    return control_decision


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=(255, 0, 0), thickness=7):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, c)


def extend_point(x1, y1, x2, y2, length):
    """ Takes line endpoints and extroplates new endpoint by a specfic length"""
    line_len = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    x = x2 + (x2 - x1) / line_len * length
    y = y2 + (y2 - y1) / line_len * length
    return x, y


def reject_outliers(data, cutoff, thresh=0.08):
    """Reduces jitter by rejecting lines based on a hard cutoff range and outlier slope """
    data = np.array(data)
    marsz = data[:, 4]
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    m = np.mean(data[:, 4], axis=0)
    # return data
    return data[(data[:, 4] <= m + thresh) & (data[:, 4] >= m - thresh)]


def merge_prev(line, prev):
    """ Extra Challenge: Reduces jitter and missed lines by averaging previous
        frame line with current frame line. """
    if prev != None:
        line = np.concatenate((line[0], prev[0]))
        x1, y1, x2, y2 = np.mean(line, axis=0)
        line = np.array([[[x1, y1, x2, y2]]], dtype=np.int32)
        return line
    else:
        return line


def separate_lines(lines):
    """ Takes an array of hough lines and separates them by +/- slope.
        The y-axis is inverted in pyplot, so the calculated positive slopes will be right
        lane lines and negative slopes will be left lanes. """
    right = []
    left = []
    for x1, y1, x2, y2 in lines[:, 0]:
        m = (float(y2) - y1) / (x2 - x1)
        if m >= 0:
            right.append([x1, y1, x2, y2, m])
        else:
            left.append([x1, y1, x2, y2, m])

    return right, left


def merge_lines(lines):
    """Merges all Hough lines by the mean of each endpoint,
       then extends them off across the image"""

    lines = np.array(lines)[:, :4]  ## Drop last column (slope)

    x1, y1, x2, y2 = np.mean(lines, axis=0)
    x1e, y1e = extend_point(x1, y1, x2, y2, -1000)  # bottom point
    x2e, y2e = extend_point(x1, y1, x2, y2, 1000)  # top point
    line = np.array([[x1e, y1e, x2e, y2e]])

    return np.array([line], dtype=np.int32)


def find_intersection(lines):
    import shapely
    from shapely.geometry import LineString, Point

    for x1, y1, x2, y2 in lines[0]:
        A = (x1, y1)
        B = (x2, y2)
    line1 = LineString([A, B])

    for x1, y1, x2, y2 in lines[1]:
        A = (x1, y1)
        B = (x2, y2)
    line2 = LineString([A, B])

    int_pt = line1.intersection(line2)
    point_of_intersection = int(int_pt.x), int(int_pt.y)
    return(point_of_intersection)


def pipeline(image, frame_number, preview=False, wind_name="test"):
    ### Params for region of interest
    # bot_left = [80, 540]
    # bot_right = [980, 540]
    # apex_right = [510, 315]
    # apex_left = [450, 315]

    control_decision = [0, 0.7, 0]
    width = image.shape[1]
    height = image.shape[0]
    bot_left = [0 + int(width*0.05), height]
    bot_right = [width, height]
    apex_right = [width - int(width*0.25), 0 + int(height*0.40)]
    apex_left = [0 + int(width*0.25), 0 + int(height*0.40)]

    v = [np.array([bot_left, bot_right, apex_right, apex_left], dtype=np.int32)]

    ### Run canny edge dection and mask region of interest
    gray = grayscale(image)
    blur = gaussian_blur(gray, 7)
    edge = canny(blur, 50, 125)
    mask = region_of_interest(edge, v)

    ### Run Hough Lines and seperate by +/- slope
    # lines = cv2.HoughLinesP(mask, 0.8, np.pi / 180, 25, np.array([]), minLineLength=50, maxLineGap=200)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 5, np.array([]), minLineLength=30, maxLineGap=400)

    right_lines, left_lines = separate_lines(lines)
    right = reject_outliers(right_lines, cutoff=(0.45, 0.75))
    right = merge_lines(right)

    left = reject_outliers(left_lines, cutoff=(-0.95, -0.45))
    left = merge_lines(left)

    lines = np.concatenate((right, left))

    inter_point = find_intersection(lines)
    control_decision = [(inter_point[0] - width/2) * 3, 0.4, 0]

    ### Draw lines and return final image
    line_img = np.copy((image) * 0)
    draw_lines(line_img, lines, thickness=10)

    line_img = region_of_interest(line_img, v)
    final = weighted_img(line_img, image)
    if frame_number % 4 == 0:
        cv2.destroyAllWindows()
        cv2.imshow(wind_name, final)
        cv2.waitKey(1)

    return control_decision