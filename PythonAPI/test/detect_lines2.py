#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import math


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


def plt_img(image, fig, axis, cmap=None):
    """ Helper for plotting images/frames """
    a = fig.add_subplot(1, 3, axis)
    imgplot = plt.imshow(image, cmap=cmap)


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

    print(point_of_intersection)
    return(point_of_intersection)


def pipeline(image, preview=False, wind_name="empty"):
    ### Params for region of interest
    # bot_left = [80, 540]
    # bot_right = [980, 540]
    # apex_right = [510, 315]
    # apex_left = [450, 315]
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

    ### Draw lines and return final image
    line_img = np.copy((image) * 0)
    draw_lines(line_img, lines, thickness=10)

    line_img = region_of_interest(line_img, v)
    final = weighted_img(line_img, image)

    # Circles on vertices corners
    cv2.circle(final, tuple(apex_left), 3, (255, 255, 0))
    cv2.circle(final, tuple(apex_right), 3, (255, 255, 0))
    cv2.circle(final, tuple(bot_left), 3, (255, 255, 0))
    cv2.circle(final, tuple(bot_right), 3, (255, 255, 0))
    # # Circle of intersection
    inter_point = find_intersection(lines)
    control_decision = [(inter_point[0] - width/2) / width, 1, 0]
    print(control_decision)
    cv2.circle(final, inter_point, 5, (255, 255, 255))

    ### Optional previwing of pipeline
    if (preview):
        cv2.destroyAllWindows()
        cv2.imshow(wind_name + str(" mask"), mask)
        cv2.imshow(wind_name + str(" edge"), edge)
        cv2.imshow(wind_name, final)
        # cv2.imshow(wind_name + str(" blur"), blur)
        cv2.waitKey(0)

    return final


# for it, img_path in enumerate(glob.glob("./imgs_lane_detection2/*.jpg")):
#     image = mpimg.imread(img_path)
#     print('This image is:', type(image), 'with dimesions:', image.shape)
#     ls = pipeline(image, preview=True)

for it, img_path in enumerate(glob.glob("./imgs_lane_detection/*.jpg")):
    image = mpimg.imread(img_path)
    image = cv2.resize(image, (960, 540))
    # cv2.imshow("test", image)
    # cv2.waitKey(0)
    # printing out some stats and plotting
    # print('This image is:', type(image), 'with dimesions:', image.shape)
    try:
        ls = pipeline(image, preview=True, wind_name=img_path)
    except Exception:
        pass
    # if it > len(glob.glob("./imgs_lane_detection/*.jpg"))-2:
    #     cv2.waitKey(0)


