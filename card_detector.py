import cv2
import numpy as np
import glob
import math
import imutils


class BlackJack_UNO:
    def __init__(self, samples, responses, thres=200, train=False):
        self.thres = thres

        try:
            self.samples = np.loadtxt(samples, np.float32)
            self.responses = np.loadtxt(responses, np.float32)
            self.responses = self.responses.reshape((self.responses.size, 1))
            self.model = cv2.ml.KNearest_create()
            self.model.train(self.samples, cv2.ml.ROW_SAMPLE, self.responses)
        except Exception:
            open(samples, 'a').close()
            open(responses, 'a').close()

    def load_data(self, data_path):
        self.data_path = data_path
        self.data_dict = {}
        for i in range(10):
            self.data_dict[str(i)] = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in glob.glob(f'{self.data_path}/{i}/*.jpg')]

    def create_data(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img_gray, self.thres, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        list_img = []
        for c in cnts:
            rect = cv2.minAreaRect(c)  # return (x,y), (w,h), angle
            x, y, w, h = cv2.boundingRect(c)
            if h/w > 1 and h > 20 and w > 20:
                x = x+int(w*0.24)
                y = y+int(h*0.25)
                h = h-int(h*0.33*2)
                w = w-int(w*0.18*2)

                cp_img = img.copy()
                num_img = cp_img[y:y+w, x:x+h]
                list_img.append(num_img)

                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x+h, y+w), (0, 0, 255), 1)

        return list_img, img

    def __crop_shape(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img_gray, self.thres, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        list_img = []
        for c in cnts:
            rect = cv2.minAreaRect(c)  # return (x,y), (w,h), angle
            x, y, w, h = cv2.boundingRect(c)
            if h/w > 1 and h > 20 and w > 20:
                x = x+int(w*0.24)
                y = y+int(h*0.25)
                h = h-int(h*0.33*2)
                w = w-int(w*0.18*2)

                list_img.append([binary[y:y+h, x:x+w], (y, x)])

                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x+h, y+w), (0, 0, 255), 1)

        return list_img, img

    def __match_shape(self, shape_list):
        return_list = []

        list_img = shape_list

        for i in list_img:
            img = i[0]
            img_temp = {}
            for num, data_path in self.data_dict.items():
                for data_img in data_path:
                    _, data_thres = cv2.threshold(data_img, self.thres, 255, cv2.THRESH_BINARY)
                    ms = cv2.matchShapes(img, data_thres, cv2.CONTOURS_MATCH_I2, 0)
                    ms = math.log10(ms)
                    if img_temp == {}:
                        img_temp['num'] = num
                        img_temp['coor'] = i[1]
                        img_temp['__ms__'] = ms
                    elif img_temp['__ms__'] > ms:
                        img_temp['num'] = num
                        img_temp['coor'] = i[1]
                        img_temp['__ms__'] = ms

            return_list.append(img_temp)

        return return_list

    def getSubImage(self, rect, src):
        # Get center, size, and angle from rect
        center, size, theta = rect

        # Convert to int
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # Get rotation matrix for rectangle
        Mo = cv2.getRotationMatrix2D(center, theta, 1)
        # Perform rotation on src image
        dst = cv2.warpAffine(src, Mo, src.shape[:2])
        out = cv2.getRectSubPix(dst, size, center)
        if theta > 50:
            out = imutils.rotate_bound(out, 90)
        return out

    def sub_img2(self, img, rect):
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        print(box)

    def train_data(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img_gray, self.thres, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        list_img = []
        roi = binary
        for c in cnts:
            rect = cv2.minAreaRect(c)  # return (x,y), (w,h), angle
            x, y, w, h = cv2.boundingRect(c)
            if h/w > 1 and h > 20 and w > 20:

                box = cv2.boxPoints(rect)
                box = np.intp(box)

                sub_img = self.getSubImage(rect, img)

                self.sub_img2(img, rect)

                hS, wS = sub_img.shape[:2]

                xS = int(wS*0.1)
                yS = int(hS*0.05)
                wS = wS-int(wS*0.70)
                hS = hS-int(hS*0.74)

                sub_gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)

                _, binS = cv2.threshold(sub_gray, self.thres, 255, cv2.THRESH_BINARY)

                roi = binS[yS:yS+hS, xS:xS+wS]

                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

        return roi, img

    def __knn_detection(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img_gray, self.thres, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi = binary
        for c in cnts:
            rect = cv2.minAreaRect(c)  # return (x,y), (w,h), angle
            x, y, w, h = cv2.boundingRect(c)
            if h/w > 1 and h > 20 and w > 20:
                # x = x+int(w*0.24)
                # y = y+int(h*0.25)
                # h = h-int(h*0.33*2)
                # w = w-int(w*0.18*2)

                # roi = binary[y:y+h, x:x+w]

                box = cv2.boxPoints(rect)
                box = np.intp(box)

                sub_img = self.getSubImage(rect, img)

                self.sub_img2(img, rect)

                hS, wS = sub_img.shape[:2]

                xS = int(wS*0.1)
                yS = int(hS*0.05)
                wS = wS-int(wS*0.70)
                hS = hS-int(hS*0.74)

                sub_gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)

                _, binS = cv2.threshold(sub_gray, self.thres, 255, cv2.THRESH_BINARY)

                roi = binS[yS:yS+hS, xS:xS+wS]

                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = self.model.findNearest(roismall, k=1)
                string = str(int((results[0][0])))

                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                # cv2.rectangle(img, (x, y), (x+h, y+w), (0, 255, 255), 1)
                cv2.putText(img, string, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

        return img

    def detect_number(self, img):

        # shape_list = self.__crop_shape(img)
        # match_data = self.__match_shape(shape_list[0])

        # for img_match in match_data:
        #     y, x = img_match['coor']
        #     cv2.putText(img, img_match['num'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # img = shape_list[1]

        img = self.__knn_detection(img)

        return img
