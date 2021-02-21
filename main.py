import cv2
import numpy as np

from card_detector import BlackJack_UNO

samples_path = 'samples.data'  # samples data (contain numpy array of feature)
responses_path = 'responses.data'  # responses data (contain classes of feature)

bj_uno = BlackJack_UNO(samples_path, responses_path, thres=220)  # create class

# create key list contains integer of waitkey keys
keys = [
    ord('1'),
    ord('2'),
    ord('3'),
    ord('4'),
    ord('5'),
    ord('6'),
    ord('7'),
    ord('8'),
    ord('9'),
    ord('s'),
    ord('u')
]

# create empty samples and responses
samples = np.empty((0, 100))
responses = []

cap = cv2.VideoCapture(2)

while cap.isOpened():

    _, img = cap.read()

    img = img[60:, :]  # cut image to make image look wider

    out = img.copy()

    out = bj_uno.detect_number(img)  # usse to detect image

    cv2.imshow('out', out)
    # cv2.imshow('roi', roi)

    wk = cv2.waitKey(10)

    if wk == ord('q'):
        break

    # roi, out = bj_uno.train_data(img) # use to train data
    # train data by key
    elif wk in keys:

        # make roi smaller for feature extraction data
        roismall = cv2.resize(roi, (10, 10))
        print(chr(wk))  # logging key

        # replace 'u' with 11 in class
        if chr(wk) == 'u':
            responses.append(int(11))
        # replace 'n' with 12 in class
        elif chr(wk) == 'n':
            responses.append(int(12))
        else:
            responses.append(int(chr(wk)))

        # reshape array to 1*100
        sample = roismall.reshape((1, 100))
        # add sample to list
        samples = np.append(samples, sample, 0)

    # save data
    elif wk == ord('s'):
        # check if overide data or append to exist data
        if bj_uno.get_new_train == False:
            last_samples = np.loadtxt('samples.data', np.float32)
            last_responses = np.loadtxt('responses.data', np.float32)
            for res in last_responses:
                responses.append(res)
            samples = np.append(samples, last_samples, 0)

        responses = np.array(responses, np.float32)
        responses = responses.reshape((responses.size, 1))
        np.savetxt('samples.data', samples)
        np.savetxt('responses.data', responses)
        print('saved')
        responses = []
        samples = np.empty((0, 100))


cap.release()
cv2.destroyAllWindows()
