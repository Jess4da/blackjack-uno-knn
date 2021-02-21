import cv2
import numpy as np


class BlackJack_UNO:
    def __init__(self, samples, responses, thres=200, new_train=False):

        self.thres = thres  # threshold number
        self.new_train = new_train  # is create new data or append exist data

        try:
            self.samples = np.loadtxt(samples, np.float32)
            self.responses = np.loadtxt(responses, np.float32)
            self.responses = self.responses.reshape((self.responses.size, 1))

            # use K-NN machine learning from cv2 built-in
            self.model = cv2.ml.KNearest_create()
            # train K-NN model
            self.model.train(self.samples, cv2.ml.ROW_SAMPLE, self.responses)
        except Exception:
            # if file not exist, create new file
            open(samples, 'a').close()
            open(responses, 'a').close()

    # get data feature
    def train_data(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img_gray, self.thres, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi = binary
        for c in cnts:
            rect = cv2.minAreaRect(c)
            x, y, w, h = cv2.boundingRect(c)
            if h/w > 1 and h > 20 and w > 20:

                box = cv2.boxPoints(rect)
                box = np.intp(box)

                x = x + int(w*0.25)
                y = y + int(h*0.25)
                w = w - int(w*0.55)
                h = h - int(h*0.45)

                roi = binary[y:y+h, x:x+w]

                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

        return roi, img

    # K-NN train detect function
    def __knn_detection(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img_gray, self.thres, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi = binary
        obj_list = []
        for c in cnts:
            rect = cv2.minAreaRect(c)
            x, y, w, h = cv2.boundingRect(c)
            if h/w > 1 and h > 20 and w > 20:

                box = cv2.boxPoints(rect)
                box = np.intp(box)

                x = x + int(w*0.25)
                y = y + int(h*0.25)
                w = w - int(w*0.55)
                h = h - int(h*0.45)

                roi = binary[y:y+h, x:x+w]

                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = self.model.findNearest(roismall, k=1)
                raw_result = results[0][0]
                if raw_result == 11:
                    string = 'u'
                elif raw_result == 12:
                    string = 'n'
                else:
                    string = str(int((raw_result)))

                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

                numObj = {
                    'coor': (x, y, w, h),
                    'num': string
                }

                obj_list.append(numObj)

                cv2.putText(img, string, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

        return img, obj_list

    # get that this is new train or append exist data
    @property
    def get_new_train(self):
        return self.new_train

    # create ui
    def create_ui(self, img, obj_list):
        iH, iW = img.shape[:2]
        iX = 0
        iY1 = int(iH*.35)
        iY2 = int(iH*.6)

        # draw line
        cv2.line(img, (iX, iY1), (iW, iY1), (0, 255, 0), 2)
        cv2.line(img, (iX, iY2), (iW, iY2), (0, 255, 0), 2)

        list_home = []
        list_away = []
        num_u = 0

        turn = 0

        for obj in obj_list:
            x, y, w, h = obj['coor']
            if y+w < iY1:
                if obj['num'] != 'u' and obj['num'] != 'n':
                    list_away.append(int(obj['num']))
                    sum_away = sum(list_away)

                elif obj['num'] == 'n' and turn == 0:
                    turn = 1

            elif y+w > iY2:
                if obj['num'] != 'u' and obj['num'] != 'n':
                    list_home.append(int(obj['num']))
                    sum_home = sum(list_home)

                else:
                    num_u += 1

        iY1_score = int(iH*.41)
        iY2_score = int(iH*.58)

        sum_home = sum(list_home)
        sum_away = sum(list_away)

        # Check win
        str_home = f'Home : {sum_home}'
        str_away = f'Away : {sum_away}'

        if sum_away == 21 or sum_home > 21:
            str_home = f'Home : {sum_home} : LOSE'
            str_away = f'Away : {sum_away} : WIN'
            turn = 2
        elif sum_home == 21 or sum_away > 21:
            str_home = f'Home : {sum_home} : WIN'
            str_away = f'Away : {sum_away} : LOSE'
            turn = 2

        if num_u > 0:
            str_home = 'Home : ? + ' + str_home[6:]

        # create 'draw' text
        if turn == 0:
            cv2.putText(img, 'Draw!', (iW - 150, iY1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        if turn == 1:
            cv2.putText(img, 'Draw!', (iW - 150, iY2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # create score text
        cv2.putText(img, str_away, (10, iY1_score), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
        cv2.putText(img, str_home, (10, iY2_score), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)

        return img

    def detect_number(self, img):

        img, obj_list = self.__knn_detection(img)

        img = self.create_ui(img, obj_list)

        return img
