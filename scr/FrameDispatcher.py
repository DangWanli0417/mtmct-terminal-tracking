# -*- coding: utf-8 -*-
# @Time: 2020/11/13 8:57
# @Author: libenchao
# @Contact: libenchao@caacsri.com
# @FileName: FrameDispatcher.py

r"""
To test the main procedure: detection -- feature extraction -- reid & tracking
"""

import cv2
import os
import redis
import time
from utils.img_utils import imageArray2ImageBytes
from threading import Thread
import numpy as np

class FrameLoader(Thread):
    def __init__(self, cameraId, videoPath, queueMax=6) -> None:
        super().__init__()
        self.cameraId = cameraId
        self.redisConn = redis.StrictRedis(host="127.0.0.1")
        self.videoCap = cv2.VideoCapture(videoPath)
        self.queueName = f"Frames_{cameraId}"
        self.queueMax = queueMax
        self.frameCount = int(self.videoCap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.imgsDir = videoPath

        # self.redisConn.delete(self.queueName)

    @property
    def redisQueueWelcome(self):
        return self.redisConn.llen(f"Frames_{self.cameraId}") <= self.queueMax

    def run(self):
        for frameId in range(0, self.frameCount, 1):
            self.videoCap.set(cv2.CAP_PROP_POS_FRAMES, frameId)
            ret, frame = self.videoCap.read()
            if ret:
                self.sentToRedis(frame)
            time.sleep(0.01)


    # def run(self):
    #     imgsList = os.listdir(self.imgsDir)
    #     imgsFileList = [fileName for fileName in imgsList if fileName.endswith('.jpg')]
    #     print(len(imgsFileList))
    #     for imgFile in imgsFileList:
    #         frame = cv2.imdecode(np.fromfile(os.path.join(self.imgsDir, imgFile), dtype=np.uint8), -1)
    #         self.sentToRedis(frame)
    #         time.sleep(0.001)




    def sentToRedis(self, datum):
        datumBytes = imageArray2ImageBytes(datum)
        self.redisConn.rpush(self.queueName, datumBytes)

    # def load(self):
    #     for frameId in range(0, self.frameCount, 15):
    #         self.videoCap.set(cv2.CAP_PROP_POS_FRAMES, frameId)
    #         ret, frame = self.videoCap.read()
    #         if ret:
    #             self.sentToRedis(frame)
    #         time.sleep(0.01)
    #
    #     print("LOAD FINISHED")


if __name__ == "__main__":
    rootPath = 'E:/WorkSpace/数据/天府图片密度测试/20210630/tmp/videopath'
    rootPath = "E:/WorkSpace/数据/detectionTestVideos"
    cameraIds = os.listdir(rootPath)
    #cameraIds = [cameraId for cameraId in cameraIds if cameraId.endswith("-0")]
    # cameraIndex2Id = {"0": "10296", "1": "10301", "2": "10313", "3": "10526",
    #                   "4": "10559", "5": "10562", "6": "10742", "7": "11385",
    #                   "8": "12239", "9": "12245", "10": "1387", "11": "1390",
    #                   "12": "1418", "13": "200", "14": "2026", "15": "2032",
    #                   "16": "2041", "17": "3327", "18": "3341", "19": "3343",
    #                   "20": "3344", "21": "3478", "22": "3954", "23": "3955",
    #                   "24": "4351", "25": "456791", "26": "598", "27": "607",
    #                   "28": "7293", "29": "7365", "30": "7411", "31": "7418",
    #                   "32": "742", "33": "7422", "34": "7443", "35": "7444",
    #                   }
    cameraIndex2Id = {"0": "TCDM_1", "1": "TCDM_2", "2":"TCDM_5", "3": "TCDM_6"}
    videoFiles = {cameraId: f"{rootPath}/{path}" for cameraId, path in enumerate(cameraIds)}
    frameLoaders = [FrameLoader(cameraIndex2Id[str(cameraId)], videoFile) for cameraId, videoFile in videoFiles.items()]


    # {'cameraId': 'TCDM_1', 'address': 'Z:/数据/detectionTestVideos/dengjikou.mp4', "Post": {"Interval": 1, "ruleId": "1", "system": "GTC","ROI": [[0, 0], [1280, 0], [1280, 720], [0, 720]]}},
    # {'cameraId': 'TCDM_2', 'address': 'Z:/数据/detectionTestVideos/dating.mp4', "Post": {"Interval": 1, "ruleId": "3", "system": "GTC","ROI": [[0, 0], [1280, 0], [1280, 720], [0, 720]]}},
    # {'cameraId': 'TCDM_5', 'address': 'Z:/数据/detectionTestVideos/quanjing.mp4', "Post": {"Interval": 1, "ruleId": "5", "system": "GTC","ROI": [[0, 0], [500, 100], [500, 720], [0, 720]]}},
    # {'cameraId': 'TCDM_6', 'address': 'Z:/数据/detectionTestVideos/zoulang.mp4', "Post": {"Interval": 1, "ruleId": "10", "system": "GTC","ROI": [[500, 500], [800, 600], [

    for frameLoader in frameLoaders:
        frameLoader.start()

    for frameLoader in frameLoaders:
        frameLoader.join()




