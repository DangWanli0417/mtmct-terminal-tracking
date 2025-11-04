# -*- coding: utf-8 -*-
# @Time: 2020/12/2 15:36
# @Author: libenchao
# @Contact: libenchao@caacsri.com
# @FileName: QueryFrameDispatcher.py

r"""
To test the main procedure: detection -- feature extraction -- reid & tracking
"""

import cv2
import os
import redis
import time
from utils.img_utils import imageArray2ImageBytes
import json
import numpy as np


class QueryFrameLoader:
    def __init__(self, dataRootPath, queueMax=4) -> None:
        super().__init__()
        self.redisConn = redis.StrictRedis()
        # folder
        self.dataRootPath = dataRootPath
        self.queueMax = queueMax
        self.queryInputQueueName = f"Query_InputImageBytes"

        # self.redisConn.delete(self.queryInputQueueName)

    def load(self):
        filesListUnderFolder = os.listdir(self.dataRootPath)
        personImagePathList = [self.dataRootPath + '/' + tmpFile for tmpFile in filesListUnderFolder if (
                    tmpFile.endswith('.jpg') or tmpFile.endswith('.JPG') or tmpFile.endswith(
                '.PNG') or tmpFile.endswith('.png'))]
        queryPersonIds = [personImagePath[:8] for personEnumId, personImagePath in enumerate(personImagePathList)]
        queryImageBytesList = [imageArray2ImageBytes(cv2.imdecode(np.fromfile(personImagePath, dtype=np.uint8), cv2.IMREAD_COLOR)) for personImagePath in personImagePathList]

        toSend = {"queryPersonIds": queryPersonIds, }
        clip = bytes(json.dumps(toSend)[1:], encoding="utf-8")
        frameBytes = bytes('{"queryImageBytesList":["', encoding="utf-8") + bytes.join(b'","', queryImageBytesList) + b'"],' + clip
        self.redisConn.rpush(self.queryInputQueueName, frameBytes)
        print("Shit")


if __name__ == "__main__":
    print("QueryFrameDispatcher")
    tmpFrameLoader = QueryFrameLoader('Z:/数据/双流机场ReID测试数据3/全身照/query')
    tmpFrameLoader.load()