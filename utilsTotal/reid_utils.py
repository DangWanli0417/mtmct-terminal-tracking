# -*- coding: utf-8 -*-
# @Time: 2020/10/28 9:43
# @Author: libenchao
# @Contact: libenchao@caacsri.com
# @FileName: reid_utils.py

import numpy as np
import cv2
from collections import defaultdict

FEATURE_GOOD = [1, 2, 3, 6, 7, 8, 10, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 69, 70, 71, 72, 73, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 110, 111, 112, 113, 115, 117, 118, 119, 120, 121, 122, 124, 125, 127, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 226, 228, 229, 230, 231, 232, 233, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 249, 250, 253, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 274, 276, 279, 280, 281, 282, 283, 284, 286, 287, 288, 289, 290, 292, 293, 296, 297, 298, 299, 300, 302, 304, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 320, 321, 324, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 337, 338, 340, 341, 342, 344, 345, 346, 347, 348, 349, 351, 352, 353, 355, 356, 357, 358, 360, 361, 362, 364, 365, 366, 367, 371, 372, 374, 375, 377, 378, 379, 380, 382, 383, 386, 388, 389, 390, 391, 392, 395, 396, 398, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 412, 413, 415, 417, 418, 419, 420, 421, 422, 423, 424, 426, 427, 428, 429, 430, 431, 433, 434, 435, 437, 438, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 454, 455, 456, 457, 458, 459, 461, 462, 463, 465, 468, 469, 470, 472, 474, 475, 476, 477, 478, 480, 482, 484, 486, 488, 489, 491, 492, 493, 495, 496, 497, 498, 499, 500, 503, 504, 506, 507, 508, 509, 510, 511]

FEATURE_COLOR = [0,  11,  13,  14,  16,  17,  23,  31,  32,  33,  34,  35,  41, 46,  49,  50,  59,  65,  67,  68,  72,  73,  76,  78,  79,  83, 84,  85,  87,  90,  91,  93,  99, 104, 105, 109, 110, 113, 115,121, 123, 127, 128, 131, 132, 134, 135, 141, 143, 144, 148, 152,156, 157, 158, 165, 167, 170, 175, 176, 178, 181, 183, 184, 191,196, 206, 212, 218, 219, 225, 228, 229, 234, 235, 236, 244, 245,252, 253, 254, 255, 256, 265, 270, 272, 273, 277, 278, 282, 289,291, 293, 297, 299, 300, 301, 303, 304, 307, 313, 320, 324, 327,329, 333, 334, 335, 336, 342, 349, 350, 352, 359, 361, 369, 372,378, 384, 386, 389, 394, 395, 398, 402, 403, 410, 412, 413, 419,424, 429, 431, 434, 441, 443, 446, 449, 455, 457, 458, 459, 462,463, 464, 472, 474, 475, 482, 484, 486, 487, 489, 494, 498, 499,501, 502, 504, 506]

__all__ = ["resize_and_pad",
           "getBetaMatrix",
           "getQueryArrayAndStdAndMeanAndBias",
           "getGalleryArrayAndStdAndMeanAndBias",
           "getPearsonArrayTrack",
           "getPearsonArrayColor",
           "ReIdMatcher",
           "getBetaHighMatrix"]


class ReIdMatcher:

    def __init__(self, reidHigh=0.3, reidLow=0.15) -> None:
        super().__init__()
        self.reidHigh = reidHigh
        self.reidLow = reidLow

    # 这里的gallery应当是trackPool的lastFeature组成的FeatureArray
    def matching(self, inputQueryArray, inputGalleryArray):
        reidBetaHighMatrix, reidBetaLowMatrix = getBetaMatrix(inputQueryArray, inputGalleryArray)
        reidBetaMatrix = np.logical_and(reidBetaHighMatrix > self.reidHigh, reidBetaLowMatrix < self.reidLow)
        reidMatchedPersonIndices, reidMatchedCurrentIndices = np.where(reidBetaMatrix == True)
        currentId2ReidMatchedPersonId, currentId2ReidMatchedScore = defaultdict(list), defaultdict(list)
        for numId, reidMatchedCurrentIndex in enumerate(reidMatchedCurrentIndices):
            currentId2ReidMatchedPersonId[reidMatchedCurrentIndex].append(int(reidMatchedPersonIndices[numId]))
            currentId2ReidMatchedScore[reidMatchedCurrentIndex].append(
                (round(reidBetaHighMatrix[reidMatchedPersonIndices[numId], reidMatchedCurrentIndex], 3),
                 round(reidBetaLowMatrix[reidMatchedPersonIndices[numId], reidMatchedCurrentIndex], 3)))
        return currentId2ReidMatchedPersonId, currentId2ReidMatchedScore


def getQueryArrayAndStdAndMeanAndBias(inputQueryArray):
    _std = np.std(inputQueryArray, axis=1,  keepdims=True)
    _mean = np.mean(inputQueryArray, axis=1, keepdims=True)
    _bias = inputQueryArray - _mean
    return inputQueryArray, _std, _mean, _bias


def getGalleryArrayAndStdAndMeanAndBias(inputGalleryArray):
    _std = np.expand_dims(np.std(inputGalleryArray, axis=1), axis=0)
    _mean = np.mean(inputGalleryArray, axis=1, keepdims=True)
    _bias = inputGalleryArray - _mean
    return inputGalleryArray, _std, _mean, _bias


def getPearsonArrayTrack(inputQueryStd, inputQueryBias, inputGalleryStd, inputGalleryBias):
    _cov = np.dot(inputQueryBias, inputGalleryBias.transpose())
    _stdArray = np.dot(inputQueryStd, inputGalleryStd)
    _corrArray: np.ndarray = _cov / _stdArray / 512
    return _corrArray

def getPearsonArrayColor(inputQueryStd, inputQueryBias, inputGalleryStd, inputGalleryBias):
    _cov = np.dot(inputQueryBias, inputGalleryBias.transpose())
    _stdArray = np.dot(inputQueryStd, inputGalleryStd)
    _corrArray: np.ndarray = _cov / _stdArray / 160
    return _corrArray


def getBetaHighMatrix(inputQueryMatrix: np.ndarray, inputGalleryMatrix: np.ndarray):
    #  ====== version v1.0
    if inputQueryMatrix.ndim == 2:
        inputQueryMatrix = np.expand_dims(inputQueryMatrix, axis=1)
    if inputGalleryMatrix.ndim == 2:
        inputGalleryMatrix = np.expand_dims(inputGalleryMatrix, axis=0)
    _maxMatrix, _minMatrix = np.maximum(inputQueryMatrix, inputGalleryMatrix), np.minimum(inputQueryMatrix, inputGalleryMatrix)
    _localSimilarityMatrix = np.tan(_minMatrix / _maxMatrix) / np.tan(1)
    np.nan_to_num(_localSimilarityMatrix, copy=False, nan=1)
    _localSimilarityMatrix = np.mean(_localSimilarityMatrix, axis=2)
    return _localSimilarityMatrix


def getBetaMatrix(inputQueryMatrix: np.ndarray, inputGalleryMatrix: np.ndarray):

    try:
        inputQueryMatrix, inputGalleryMatrix = inputQueryMatrix[:, FEATURE_GOOD], inputGalleryMatrix[:, FEATURE_GOOD]
    except TypeError:
        print("shit...")


    if inputQueryMatrix.ndim == 2:
        inputQueryMatrix = np.expand_dims(inputQueryMatrix, axis=1)
    if inputGalleryMatrix.ndim == 2:
        inputGalleryMatrix = np.expand_dims(inputGalleryMatrix, axis=0)
    _maxMatrix, _minMatrix = np.maximum(inputQueryMatrix, inputGalleryMatrix), np.minimum(inputQueryMatrix, inputGalleryMatrix)
    _localSimilarityMatrix = _minMatrix / _maxMatrix
    np.nan_to_num(_localSimilarityMatrix, copy=False, nan=1.0)
    # _localSimilarityMatrixBefore[_localSimilarityMatrixBefore > 0.9] = 1
    highMatrix = np.sum(_localSimilarityMatrix > 0.5, axis=2) / len(FEATURE_GOOD)
    lowMatrix = np.sum(np.exp(_minMatrix) / np.exp(_maxMatrix) < 0.1, axis=2) / len(FEATURE_GOOD)
    return highMatrix, lowMatrix


def resize_and_pad(_img: np.ndarray, w: int=128, h: int=256):
    ratio = _img.shape[0] / _img.shape[1]
    zeros = np.zeros((256, 128), dtype=np.uint8)
    if ratio < 2:
        resizedHeight = int(ratio * 128)
        _img = cv2.resize(_img, (128, resizedHeight))
        pad_up = (256 - resizedHeight) // 2
        pad_bottom = 256 - pad_up - resizedHeight
        #background_img[pad_up:resizedHeight + pad_up, :,:] = _img
        _img = cv2.copyMakeBorder(_img, pad_up, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(104, 116, 124))
    else:
        resizedWidth = int(256 / ratio)
        _img = cv2.resize(_img, (resizedWidth, 256))
        pad_left = (128 - resizedWidth) // 2
        pad_right = 128 - pad_left - resizedWidth
        _img = cv2.copyMakeBorder(_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(104, 116, 124))
        #background_img[:,pad_left:resizedWidth + pad_left,:] = _img
    return _img


# def resize_and_pad(_img: np.ndarray,background_img:np.ndarray, w: int=128, h: int=256):
#     ratio = _img.shape[0] / _img.shape[1]
#     zeros = np.zeros((256, 128), dtype=np.uint8)
#     if ratio < 2:
#         resizedHeight = int(ratio * 128)
#         _img = cv2.resize(_img, (128, resizedHeight))
#         pad_up = (256 - resizedHeight) // 2
#         pad_bottom = 256 - pad_up - resizedHeight
#         #background_img[pad_up:resizedHeight + pad_up, :,:] = _img
#         _img = cv2.copyMakeBorder(_img, pad_up, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(104, 116, 124))
#     else:
#         resizedWidth = int(256 / ratio)
#         _img = cv2.resize(_img, (resizedWidth, 256))
#         pad_left = (128 - resizedWidth) // 2
#         pad_right = 128 - pad_left - resizedWidth
#         _img = cv2.copyMakeBorder(_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(104, 116, 124))
#         #background_img[:,pad_left:resizedWidth + pad_left,:] = _img
#     return _img

