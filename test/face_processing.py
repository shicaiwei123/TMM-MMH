import cv2
import numpy as np
import shutil
import os
import sys
sys.path.append('..')
from lib.processing_utils import LandmarksDetection, get_file_list, makedir


def recut_face_with_landmarks(origin_face_dir, target_face_dir, display=False):
    '''
    检测人脸关键点,利用关键点裁剪人脸,
    :param origin_face_dir:
    :param target_face_dir:
    :return:
    '''
    makedir(target_face_dir)
    landmarks_detector = LandmarksDetection()
    x_data = np.zeros(27)
    y_data = np.zeros(27)
    file_list = get_file_list(origin_face_dir)
    for file_path in file_list:
        img = cv2.imread(file_path)
        h, w = img.shape[0], img.shape[1]
        mask = np.zeros((h, w), dtype=np.uint8)
        face_landmarks = landmarks_detector.landmarks_detect(img, display=display)
        face_landmarks = np.array(face_landmarks)
        x_temp = face_landmarks[0:17, 0]
        y_temp = face_landmarks[0:17, 1]
        x_data[0:17] = x_temp
        y_data[0:17] = y_temp
        for i in range(5):
            a = face_landmarks[26 - i][0]
            x_data[17 + i] = face_landmarks[26 - i][0]
            y_data[17 + i] = face_landmarks[26 - i][1]

        for i in range(5):
            x_data[22 + i] = face_landmarks[22 - i][0]
            y_data[22 + i] = face_landmarks[22 - i][1]

        pts = np.vstack((x_data, y_data)).astype(np.int32).T
        cv2.fillConvexPoly(mask, pts, (255), 8, 0)
        result = cv2.bitwise_and(img, img, mask=mask)

        if display:
            cv2.imshow("mask", mask)
            # 根据mask，提取ROI区域
            cv2.imshow("result", result)
            cv2.waitKey(0)

        save_path = os.path.join(target_face_dir, file_path.split('/')[-1])
        print(save_path)

        flag=cv2.imwrite(save_path,result)
        print(flag)
        # shutil.copy(result, save_path)


if __name__ == '__main__':
    origin_face_dir = "/home/bbb/shicaiwei/data/liveness_data/CASIA-FASD/CASIA-FASD_face/test/spoofing"
    target_face_dir = "/home/bbb/shicaiwei/data/liveness_data/CASIA-FASD/CASIA-FASD_face_landmarks/test/spoofing"
    recut_face_with_landmarks(origin_face_dir=origin_face_dir, target_face_dir=target_face_dir)
