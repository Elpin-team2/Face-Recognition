import cv2
import os
import glob
from random import randint


# 네트워크 아키텍처에 대한 텍스트 설명 파일
# faceProto_path = "./opencv_face_detector.pbtxt"
# 훈련모델
# faceModel_path = "./opencv_face_detector_uint8.pb"

class Img2face:
    def __init__(self, faceModel_path, faceProto_path):
        self.faceModel_path = faceModel_path
        self.faceProto_path = faceProto_path


    # 설진영님 흑백얼굴사진 추출 함수
    def getFaceBox2(self, frame, conf_threshold=0.7):
        # Load network
        faceNet = cv2.dnn.readNet(self.faceModel_path, self.faceProto_path)
        faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # DNN_TARGET_CPU
        # CPU로 하면 오래 걸림

        frameOpencvDnn = frame.copy()
        frameWidth = frameOpencvDnn.shape[1]
        frameHeight = frameOpencvDnn.shape[0]

        blob = cv2.dnn.blobFromImage(cv2.resize(frameOpencvDnn,(300, 300)), 1.0, (300, 300), [104, 117, 123], True, False)
        faceNet.setInput(blob)
        detections = faceNet.forward()
        bboxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                # 0 ~ 30랜덤 값
                ransize = randint(0, 30)
                # 랜덤하게 얼굴좌표 뽑아줌
                dst = frameOpencvDnn[y1-ransize:y2+ransize,x1-ransize:x2+ransize]
                gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
        
        return gray



    # 이미지 회전, 확대, 축소, 반전 랜덤하게 설정해주는 함수
    def img_rotation(self, img_bgr):
        # -10 ~ 10 회전 각도
        angle = randint(-100, 100)*0.1
        # 75% ~ 125% 이미지 크기 변경
        # scale = randint(75, 125)*0.01
        # 50% 반전
        flip = randint(0, 1)
        
        # 50% 확률로 좌우 반전
        if flip == 1:
            img_bgr = cv2.flip(img_bgr, 1)

        # 이미지 좌표 뽑아내기
        rows, cols = img_bgr.shape[:2]
        # 이미지 중간을 중심으로 회전, 확대 또는 축소
        rotate = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        # 이미지에 변경사항 적용
        img_rotate = cv2.warpAffine(img_bgr, rotate, (cols, rows))

        return img_rotate
    # 출처: https://ansan-survivor.tistory.com/641 [안산드레아스]



def img_face(path):
    people_dict = {}
    for person in os.listdir(path):
        print('loading person: ' + person)
        # glob을 이용해서 각 이름의 jpg파일 빼오기
        img_bgr = [cv2.imread(file) for file in glob.glob(f"/content/data/lfw_funneled/{person}/*.jpg")]
        # 사진 랜덤하게 회전, 크기 조절, 반전해주는 함수
        img_rotate = [img_rotation(img) for img in img_bgr]
        try:
            # 흑백얼굴사진 추출
            gray_face = [getFaceBox2(face) for face in img_rotate]
        except Exception as e:
            # 얼굴 추출 실패
            print('얼굴인식 실패: ', e)
            continue
        # 딕셔너리에 얼굴 사진 추가
        people_dict[person] = gray_face
        # break
            
    return people_dict
    # 이미지 -> numpy.ndarray로 딕셔너리에 저장