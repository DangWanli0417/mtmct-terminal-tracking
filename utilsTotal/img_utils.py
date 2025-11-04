# -*- coding: utf-8 -*-
# @Time: 2020/10/29 9:16
# @Author: libenchao
# @Contact: libenchao@caacsri.com
# @FileName: img_utils.py

import numpy as np
import cv2
import base64
SAMPLE_RATIO = 5
QUALITY_INDEX = {".jpg": int(cv2.IMWRITE_JPEG_QUALITY),
                 ".webp": int(cv2.IMWRITE_WEBP_QUALITY)}

__all__=['lowlight_process',
         'imageArray2ImageBytes',
         'imageBytes2ImageArray',
         'compressedFeatureBytes2Array',
         'compressedFeatureTuple2Array',
         'yx2xy',
         'drawBbox',
         'imageArray2ImageString']


def imageArray2ImageBytes(imgArray, form=".webp", quality=10):
    _frame: np.ndarray = cv2.imencode(form, imgArray, [QUALITY_INDEX[form], quality])[1]
    _frame: bytes = _frame.tobytes()
    _frame = base64.b64encode(_frame)
    return _frame


def imageArray2ImageString(imgArray, form=".webp", quality=10):
    _frame: np.ndarray = cv2.imencode(form, imgArray, [QUALITY_INDEX[form], quality])[1]
    _frame: str = base64.b64encode(_frame.tostring()).decode()
    return _frame


def imageBytes2ImageArray(imgBytes):
    imgData = cv2.imdecode(np.frombuffer(base64.b64decode(imgBytes), dtype="uint8"), flags=cv2.IMREAD_COLOR)
    return imgData


def compressedFeatureBytes2Array(featureBytes):
    if featureBytes is None:
        return None
    featureData = cv2.imdecode(np.frombuffer(base64.b64decode(featureBytes), dtype="uint8"), flags=cv2.IMREAD_GRAYSCALE)
    return featureData


def compressedFeatureTuple2Array(featureBytes, arrayMin, arrayMax):
    if featureBytes is None:
        return None
    featureData = compressedFeatureBytes2Array(featureBytes) / 255.0 * (arrayMax - arrayMin) + arrayMin
    return featureData


def detect_lowlight(img, luma_threshold) -> bool:
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    width, height = img.shape
    ratio_w = width / SAMPLE_RATIO
    ratio_h = height / SAMPLE_RATIO
    luma_count = 0
    for i in range(0, height, ratio_h):
        for j in range(0, width, ratio_w):
            luma_count += img_yuv[i, j, 0]

    pic_luma = luma_count / (SAMPLE_RATIO * SAMPLE_RATIO)
    if (pic_luma < luma_threshold):
        return True
    else:
        return False


def VRD_EHN_CLIP(v):
    if v < 0:
        return 0
    elif v > 255:
        return 255
    else:
        return v


def getLut() -> np.ndarray:
    maxlut = np.zeros((256, 256), dtype=np.float32)
    for x in range(256):
        for y in range(256):
            t = (0.044 + 0.0039 * y) ** 0.7
            temp = VRD_EHN_CLIP((x - 3.0) / t + 3.0) / 255.0
            maxlut[x, y] = VRD_EHN_CLIP(temp ** 1.22) * 255.0

    return maxlut


def lowlight_process(img: np.ndarray):
    """
    :param img: bgr image
    :return: bgr image
    """
    # img : yuv image
    height, width, _ = img.shape
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    enhance_matrix = img_yuv[:, :, 0].copy()
    maxlut = getLut()
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            enhance_matrix[i, j] = VRD_EHN_CLIP(
                0.85 * (img_yuv[i - 1, j, 0] + img_yuv[i, j - 1, 0] + img_yuv[i, j,0] * 4 + img_yuv[i + 1, j,0] + img_yuv[i, j + 1,0]) / 8.0)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            img_yuv[i, j, 0] = maxlut[img_yuv[i, j, 0]][enhance_matrix[i, j]]
            img_yuv[i, j, 0] = VRD_EHN_CLIP(0.6 * img_yuv[i, j, 0] + 0.4 * enhance_matrix[i][j])

    img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_bgr

def yx2xy(rois_yx:list) -> list:
    rois_xy = [[roi[1], roi[0], roi[3], roi[2]] for roi in rois_yx]
    return rois_xy

def drawBbox(rois:list, frame:np.ndarray)->np.ndarray:
    for coord in rois:
        tl = 1
        color = (0, 255, 0)
        c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
        cv2.rectangle(frame, c1, c2, color, thickness=tl)
    return frame