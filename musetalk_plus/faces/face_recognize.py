import cv2
import numpy as np
import face_recognition


class FaceRecognizer:

    @staticmethod
    def face_locations(image):
        if isinstance(image, str):
            image = face_recognition.load_image_file(image)
        locations = face_recognition.face_locations(image)
        if len(locations) == 0:
            return None
        return locations[0]

    @staticmethod
    def face_landmarks(image, face_locations=None):
        landmark_list = face_recognition.face_landmarks(image, face_locations)
        if len(landmark_list) == 0:
            return None
        landmark = landmark_list[0]
        # 鼻尖和上嘴唇
        nose_bridge = landmark['nose_bridge']
        nose_tip = landmark['nose_tip']
        chin = landmark['chin']
        nose_bottom_y = (max(nose_bridge, key=lambda x: x[-1])[-1] + max(nose_tip, key=lambda x: x[-1])[-1]) // 2
        # 找到从鼻子下面到下巴的轮廓
        cbn = np.array([point for point in chin if point[1] > nose_bottom_y], dtype=np.int32)
        landmark_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(landmark_mask, [cbn], -1, (255, 255, 255), -1)
        return landmark_mask


if __name__ == '__main__':
    fr = FaceRecognizer()
    img = cv2.imread('00000002.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    from PIL import Image

    Image.fromarray(fr.face_landmarks(image=img, face_locations=None)).show()
