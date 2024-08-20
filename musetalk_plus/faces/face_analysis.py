import cv2
import numpy as np
from PIL import Image, ImageDraw
from mmpose.apis import inference_topdown, init_model


class FaceAnalyst:

    def __init__(self, config_path, model_path):
        self.model = init_model(config_path, model_path)

    def analysis(self, image: str):
        results = inference_topdown(self.model, image)
        if len(results) == 0:
            return None
        key_points = results[0].pred_instances['keypoints']
        return key_points

    @staticmethod
    def face_landmark_mask(image_size: [int, int], key_points):
        # 选择下半脸的关键点
        landmark_points = key_points[0][23:91]
        lower_half_face = landmark_points[2:15].astype(np.int32)
        face_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        # 创建一个平滑的弧线
        curve_points = FaceAnalyst.create_smooth_curve(lower_half_face)
        # 生成mask
        cv2.fillPoly(face_mask, [curve_points], (255, 255, 255))
        face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
        return face_mask

    @staticmethod
    def create_smooth_curve(points):
        # 使用OpenCV的approxPolyDP方法来平滑点，生成弧线
        curve = cv2.approxPolyDP(points, epsilon=1, closed=False)
        return curve

    @staticmethod
    def face_location(key_points, shift=15):
        landmark_points = key_points[0][23:91]
        face_area = landmark_points[0: 27]
        min_x = np.min(face_area[:, 0])
        min_y = np.min(face_area[:, 1])
        max_x = np.max(face_area[:, 0])
        max_y = np.max(face_area[:, 1])
        return int(min_x), int(min_y - shift), int(max_x), int(max_y)


if __name__ == '__main__':
    config_file = r'F:\Workplace\MuseTalkPlus\musetalk\utils\dwpose\rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    checkpoint_file = r'F:\Workplace\MuseTalkPlus\models\dwpose\dw-ll_ucoco_384.pth'
    fa = FaceAnalyst(config_file, checkpoint_file)
    pts = fa.analysis('00000002.png')
    bbox = fa.face_location(pts)
    im = Image.open('00000002.png')
    ImageDraw.Draw(im).rectangle(bbox, width=2)
    im.show()
