#example script to detect traffic lights

im_res = image.copy()
        print("*")
        frame_rgb_split = cv2.split(image)
        frame_rgb_sum = np.uint16(frame_rgb_split[0]) + np.uint16(frame_rgb_split[1]) + np.uint16(frame_rgb_split[2])

        # frame_rgb_split[0] = np.uint8(255 * np.true_divide(frame_rgb_split[0], frame_rgb_sum))
        # frame_rgb_split[1] = np.uint8(255 * np.true_divide(frame_rgb_split[1], frame_rgb_sum))
        # frame_rgb_split[2] = np.uint8(255 * np.true_divide(frame_rgb_split[2], frame_rgb_sum))


        frame_rgb_normalized = cv2.merge(frame_rgb_split)
        #cv2.imshow('NAnana', frame_rgb_normalized)

        #Red
        mask_r = np.uint8(frame_rgb_split[2] > 180)
        mask_g = np.uint8(frame_rgb_split[1] >= 0) & np.uint8(frame_rgb_split[1] < 100)
        mask_b = np.uint8(frame_rgb_split[0] < 120)
        mask1 = mask_r & mask_g & mask_b

        #Yellow
        # mask_r = np.uint8(frame_rgb_split[2] > 68)
        # mask_g = np.uint8(frame_rgb_split[1] > 50)
        # mask_b = np.uint8(frame_rgb_split[0] < 50)
        # mask2 = mask_r & mask_g & mask_b

        #Green
        # mask_r = np.uint8(frame_rgb_split[2] < 40)
        # mask_g = np.uint8(frame_rgb_split[1] > 70)
        # mask_b = np.uint8(frame_rgb_split[0] > 70)
        # mask3 = mask_r & mask_g & mask_b

        mask = mask1 #| mask3

        candidates = np.uint8(frame_rgb_normalized)

        candidates[mask == 0] = 0

        candidates_gray = cv2.cvtColor(candidates, cv2.COLOR_RGB2GRAY)
        candidates_rgb = cv2.split(candidates)

        kernel = np.ones((3,3), np.uint8)
        candidates_rgb[0] = cv2.erode(candidates_rgb[0], kernel, iterations=1)
        candidates_rgb[1] = cv2.erode(candidates_rgb[1], kernel, iterations=1)
        candidates_rgb[2] = cv2.erode(candidates_rgb[2], kernel, iterations=1)

        kernel = np.ones((5,5), np.uint8)
        candidates_rgb[0] = cv2.dilate(candidates_rgb[0], kernel, iterations=1)
        candidates_rgb[1] = cv2.dilate(candidates_rgb[1], kernel, iterations=1)
        candidates_rgb[2] = cv2.dilate(candidates_rgb[2], kernel, iterations=1)

        thresh, candidates_rgb[0] = cv2.threshold(candidates_rgb[0], 1, 255, cv2.THRESH_BINARY)
        thresh, candidates_rgb[1] = cv2.threshold(candidates_rgb[1], 1, 255, cv2.THRESH_BINARY)
        thresh, candidates_rgb[2] = cv2.threshold(candidates_rgb[2], 1, 255, cv2.THRESH_BINARY)

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidates_rgb[0] | candidates_rgb[1] | candidates_rgb[2])

        #return [candidates_rgb[0] | candidates_rgb[1] | candidates_rgb[2], [None, 0, 0]]
        #cv2.imshow('Biforr', candidates_rgb[0] | candidates_rgb[1] | candidates_rgb[2])
        print(nlabels)
        for i in range(1, nlabels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            # if abs(h - w) > 0.25 * w:
            #     (candidates_rgb[1])[labels == i] = 0
            #     continue
            # if abs(h - w) > 0.25 * h:
            #     (candidates_rgb[1])[labels == i] = 0
            #     continue
            # if stats[i, cv2.CC_STAT_AREA] < 20:
            #     (candidates_rgb[1])[labels == i] = 0
            #     continue

            one_object = candidates_rgb[0] | candidates_rgb[1] | candidates_rgb[2]
            one_object[labels != i] = 0
            thresh, one_object = cv2.threshold(one_object, 0, 255, 0)
            # im2, contours, hierarchy = cv2.findContours(np.uint8(one_object), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours, hierarchy = cv2.findContours(np.uint8(one_object), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 1:
                # (candidates_rgb[1])[labels == i] = 0
                continue

            cv2.drawContours(im_res, contours, -1, (255, 0, 0), 5)
        return [im_res, [None, 0, 0]



#example script to detect lane


        imshape = image.shape
        control_decision = [0, 0, 0]
        lower_left = [0, imshape[0] - imshape[0]//9] # do dopracowania
        lower_right = [imshape[1], imshape[0]- imshape[0]//9]
        top_left = [imshape[1] // 2 - imshape[1] // 5, imshape[0] // 5 * 3]  # - imshape[0] // 5]
        top_right = [imshape[1] // 2 + imshape[1] // 5, imshape[0] // 5 * 3]  # - imshape[0] // 5]

        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)[:, :, 2]
        # im_bin = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY_INV)

        height, width = im_gray.shape

        im_res = np.zeros((height, width, 1), np.uint8)

        im_gray_ = cv2.blur(im_gray, (5, 5))

        im_line = cv2.Canny(im_gray_, 60, 120)
        vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
        im_line = self.region_of_interest(im_line, vertices)

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
        try:
            [a1, b1] = [sum(l1)/len(l1), sum(l2)/len(l2)]
            [a2, b2] = [sum(r1)/len(r1), sum(r2)/len(r2)]
            common_point = (int(a1*((b2-b1)/(a1-a2))+b1), int((b2-b1)/(a1-a2)))
            # print(common_point[0] - image.shape[1]//2)
            cv2.circle(image, common_point, 10, (255, 255, 255))
            control_decision = [(common_point[0] - image.shape[1]//2) * 50, 50000, 64000]
            # self._outq_control.put([(common_point[0] - image.shape[1]//2) * 50, 50000, 64000])
        except ZeroDivisionError:
            pass

        # print(common_point)
        # cv2.imshow('Original', im)
        # cv2.imshow('Lines', im_line)
        # # cv2.imshow('Hough', im_hough)
        # cv2.imshow('Binary', im_bin)
        # im_res = cv2.bitwise_and(im_res, cv2.dilate(im_line, (5, 5)))
        process_time = time.time() - start_time

        #print(process_time)
        return [image, control_decision]
        return [im_gray, control_decision]



#example script to detect car
        self.get_speed()
        if self.pause == 1:
            return [image, [None, 0, 0]]
        #detekcja aut start
        im = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        # cv2.imshow('Image', im)

        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        canny = cv2.Canny(im, 150, 200)

        # cv2.imshow('Canny', canny)

        height, width = im.shape

        min_start_x = 300
        width_x = 75
        min_diff = np.Inf
        min_diff_img = None
        img_L = None
        img_R = None

        res = np.zeros((height, width), np.float)

        for start_x in range(0, width-150):

            cannyL = canny[:, start_x:start_x+width_x]
            cannyR = canny[:, start_x+width_x:start_x+2*width_x]
            cannyL = cannyL[:, ::-1]

            #diff_img = cv2.absdiff(cannyR, cannyL)
            kernel = np.ones((3, 1))
            cannyL_dilated = cv2.dilate(cannyL, kernel)

            sym1 = cv2.bitwise_and(cannyR, cannyL_dilated)/255 * 3
            sym2 = cv2.bitwise_xor(cannyL, cannyR)/255 * -1
            sym = sym1 + sym2

            diffs = np.sum(sym, 1)
            #diffs = np.sum(diffs, 1)
            diffs2 = []
            rect_h = 15
            for i in range(0, rect_h):
                diffs2.append(0)

            for i in range(rect_h, len(diffs) - rect_h):
                sum_temp = 0
                for j in range(-rect_h, rect_h + 1):
                    sum_temp = sum_temp + diffs[i + j]
                diffs2.append(sum_temp)

            for i in range(0, rect_h):
                diffs2.append(0)

            res[:, start_x+width_x] = diffs2
            #diff = np.sum(diff_img)
            #if min_diff > diff:
            #    min_diff = diff
             #   min_diff_img = diff_img.copy()
            #    img_L = cannyL.copy()
            #    img_R = cannyR.copy()



        maxima_per_line = 1
        for j in range(0, maxima_per_line):
            maxima = np.argmax(res, 1)
            for i in range(0, len(maxima)):
                im[i, maxima[i]] = 255
                res[i, maxima[i]] = 0
        process_time = time.time() - start_time

        #print(process_time)
        return [im, [None, 0, 0]]