import cv2
import os
import glob
import numpy as np
from card_detector import BlackJack_UNO


# cv2.namedWindow('GUI')
# cv2.createTrackbar('CardNumber', 'GUI', 0, 9, lambda t: None)

# data_path = 'new data'
samples_path = 'samples.data'
responses_path = 'responses.data'


bj_uno = BlackJack_UNO(samples_path, responses_path, thres=150)
# bj_uno.load_data(data_path)

keys = [i for i in range(48, 58)]
responses = []
samples = np.empty((0, 100))

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():

    _, img = cap.read()

    img = img[:, 215:]

    out = img.copy()

    # list_img, out = bj_uno.create_data(img)
    # card_number = cv2.getTrackbarPos('CardNumber', 'GUI')

    # roi, out = bj_uno.train_data(img)

    out = bj_uno.detect_number(img)

    cv2.imshow('out', out)

    wk = cv2.waitKey(10)

    if wk == ord('q'):
        break
    elif wk in keys:
        roismall = cv2.resize(roi, (10, 10))
        print(chr(wk))

        responses.append(int(chr(wk)))
        sample = roismall.reshape((1, 100))
        samples = np.append(samples, sample, 0)

        # roismall_flip = cv2.flip(roismall, -1)
        # responses.append(int(chr(wk)))
        # sample_flip = roismall_flip.reshape((1, 100))
        # samples = np.append(samples, sample_flip, 0)

    elif wk == ord('s'):
        responses = np.array(responses, np.float32)
        responses = responses.reshape((responses.size, 1))
        np.savetxt('samples.data', samples)
        np.savetxt('responses.data', responses)
        break
# elif wk == ord('c'):
    #     str_num = str(card_number)
    #     try:
    #         os.mkdir(data_path)
    #     except FileExistsError:
    #         pass
    #     os.chdir(data_path)
    #     try:
    #         os.mkdir(str_num)
    #     except FileExistsError:
    #         pass
    #     for img in list_img:
    #         count = 0
    #         for i in glob.glob(f'{str_num}/*.jpg'):
    #             count += 1
    #         print(f'write {str_num}')
    #         cv2.imwrite(f'{str_num}/{count+1}.jpg', img)
    #         flip_img = cv2.flip(img, 0)
    #         cv2.imwrite(f'{str_num}/{count+2}.jpg', flip_img)

    #     os.chdir('..')


cap.release()
cv2.destroyAllWindows()
