# -*- coding: utf-8 -*-
# @Time: 2020/11/11 11:21
# @Author: libenchao
# @Contact: libenchao@caacsri.com
# @FileName: PersonFeatureExtractorLauncher.py
import cv2
import numpy as np
import torch
from abc import abstractmethod
from ReidNetwork import osnet_ain_x1_0
from utils.reid_utils import resize_and_pad
import redis
import json
from utils.img_utils import imageBytes2ImageArray, imageArray2ImageBytes, compressedFeatureBytes2Array, imageArray2ImageString
from collections import defaultdict, deque
from threading import Thread
import os
import time
import encodings.idna

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

with open('./configs/DefaultConfigs.json') as fileReader:
    jsonData = json.load(fileReader)

REDIS_CONN_HOST = jsonData["FeatureExtractorRedisConnHost"]
REDIS_CONN_PORT = jsonData["FeatureExtractorRedisConnPort"]
REDIS_SEND_CONN_HOST = jsonData["FeatureExtractorRedisSendConnHost"]
REDIS_SEND_CONN_PORT = jsonData["FeatureExtractorRedisSendConnPort"]
PIXEL_THRESHOLD = jsonData["PixelThreshold"]


class FeatureExtractorWrapper:
    def __init__(self, registerDetectionProcessIds, cameraIds, deviceId, batchSize) -> None:
        super().__init__()

        # 外部连接配置
        self.redisConn = redis.StrictRedis(host=REDIS_CONN_HOST, port=REDIS_CONN_PORT)
        self.redisSendConn = redis.StrictRedis(host=REDIS_SEND_CONN_HOST, port=REDIS_SEND_CONN_PORT)

        # 基础设置
        self.device = torch.device(deviceId)
        self.cameraIds = cameraIds
        self.batchSize = batchSize
        self.registerDetectionProcessIds = registerDetectionProcessIds
        self.detectionResultQueueNames = {pid: f"Detection_{pid}" for pid in self.registerDetectionProcessIds}
        self.galleryResultQueueNames = {pid: f"Gallery_ReidFeature_{pid}" for pid in self.registerDetectionProcessIds}

        # read camera configs
        self.cameraConfigs = {}
        self.securityCameras = set()
        self.wholeBodyCameras = set()
        for cameraId in self.cameraIds:
            if os.path.exists(f"./configs/{cameraId}.json"):
                with open(f"./configs/{cameraId}.json") as f:
                    if cameraId in self.cameraConfigs:
                        self.cameraConfigs[cameraId] = json.load(f)
        if os.path.exists("./configs/CameraRelation.json"):
            with open("./configs/CameraRelation.json") as f:
                self.cameraRelation = json.load(f)
            for securityCamera, wholeBodyCamera in self.cameraRelation.items():
                self.securityCameras.add(securityCamera)
                self.wholeBodyCameras.add(wholeBodyCamera)

        self.queryInputQueueName = f"Query_InputImageBytes"
        self.queryResultQueueName = f"Query_ReidFeature"

        # data threading
        self.dataPackageReceiveQueue = deque(maxlen=5)
        self.dataPackageReceiveEnsembleThread = Thread(target=self.dataPackageEnsemble)
        self.dataPackageReceiveEnsembleThread.start()

        # data threading
        self.dataPackageSendQueue = deque(maxlen=5)
        self.dataPackageSendThread = Thread(target=self.dataPackageSend)
        self.dataPackageSendThread.start()

    @abstractmethod
    def preProcess(self, data):
        pass

    @abstractmethod
    def poseProcess(self, data):
        pass

    def add2SendQueue(self, data):
        self.dataPackageSendQueue.append(data)

    @abstractmethod
    def extractFeatureData(self) -> np.ndarray:
        pass

    @abstractmethod
    def extractQueryBatchFeature(self, imageBatch: list) -> list:
        pass

    @abstractmethod
    def extractQueryListFeature(self, imageList):
        pass

    def doMyJob(self):
        # 更新frame行人检测相关数据域，并准备好计算所需临时数据
        self.processQueryJob()
        toSendPackage = self.extractFeatureData()
        if toSendPackage is not None:
            self.add2SendQueue(toSendPackage)

    @staticmethod
    def personSizeJudge(personLocation, pixelThreshold: int = PIXEL_THRESHOLD):
        return (personLocation[3] - personLocation[1]) * (personLocation[2] - personLocation[0]) >= pixelThreshold

    def dataPackageEnsemble(self):
        while True:
            # 从外部连接加载运算基础数据
            for detectionQueueId, detectionResultQueueName in self.detectionResultQueueNames.items():
                if len(self.dataPackageReceiveQueue) == self.dataPackageReceiveQueue.maxlen:
                    time.sleep(0.01)
                    continue
                detectionResultRawData = self.redisConn.lpop(detectionResultQueueName)
                if detectionResultRawData is None:
                    time.sleep(0.01)
                    continue
                parsedData = json.loads(detectionResultRawData)
                frameBytesList, frameUnFinedLocationsList, frameCamerasList = parsedData["frameBytesList"], parsedData["batchLocationsList"], parsedData["cameras"]
                frameArrayList = []
                for frameIndex, frameBytes in enumerate(frameBytesList):
                    configFlag = frameCamerasList[frameIndex] in self.cameraConfigs
                    originalFrameArray = imageBytes2ImageArray(frameBytes)
                    if configFlag:
                        cameraROI = self.cameraConfigs[frameCamerasList[frameIndex]]["InputRegion"]
                        frameArrayList.append(originalFrameArray[cameraROI[0]:cameraROI[2], cameraROI[1]:cameraROI[3]])
                    else:
                        frameArrayList.append(originalFrameArray)

                frameTimeStampList = parsedData["frameTimeStampList"]
                # 对行人照片进行crop
                # 先清空数据暂存区
                personNumber = 0
                personImageList = []
                frameFinedLocationsList = []
                frameFeatureArrayIndexList = []
                frameExtraFeatureArrayIndexList = []
                # for store half body image
                extraPersonImageList = []
                extraFrameIndex = []
                # 按帧进行迭代
                for frameIndex, frameArray in enumerate(frameArrayList):
                    frameLocations = frameUnFinedLocationsList[frameIndex]
                    frameFinedLocations = []
                    frameArrayIndexStart = personNumber

                    # 对帧中的每一个人进行迭代
                    for personIndex, personLocation in enumerate(frameLocations):

                        # 按像素大小过滤
                        if not self.personSizeJudge(personLocation):
                            continue
                        frameFinedLocations.append(personLocation)

                        cameraId = frameCamerasList[frameIndex]
                        personImageList.append(resize_and_pad(frameArray[personLocation[1]: personLocation[3], personLocation[0]:personLocation[2]]))
                        personNumber += 1

                    frameArrayIndexEnd = personNumber
                    frameExtraArrayIndexStart = frameArrayIndexEnd
                    # 对帧中的每一个人进行迭代 extra
                    for personIndex, personLocation in enumerate(frameLocations):
                        # 按像素大小过滤
                        if not self.personSizeJudge(personLocation):
                            continue
                        cameraId = frameCamerasList[frameIndex]
                        if cameraId in self.wholeBodyCameras:
                            personImageList.append(resize_and_pad(frameArray[personLocation[1]: int((personLocation[1] + personLocation[3]) / 2), personLocation[0]:personLocation[2]]))
                            personNumber += 1
                    frameExtraArrayIndexEnd = personNumber

                    frameFinedLocationsList.append(frameFinedLocations)
                    frameFeatureArrayIndexList.append((frameArrayIndexStart, frameArrayIndexEnd))
                    frameExtraFeatureArrayIndexList.append((frameExtraArrayIndexStart, frameExtraArrayIndexEnd))

                processedPersonImageList = self.preProcess(personImageList)

                # 更新数据包字段
                parsedData["processedPersonImageList"] = processedPersonImageList
                parsedData["detectionQueueId"] = detectionQueueId
                parsedData["frameExtraFeatureArrayIndexList"] = frameExtraFeatureArrayIndexList
                parsedData["frameFinedLocationsList"] = frameFinedLocationsList
                parsedData["frameFeatureArrayIndexList"] = frameFeatureArrayIndexList
                parsedData["frameExtraFeatureArrayIndexList"] = frameExtraFeatureArrayIndexList
                parsedData["personImageList"] = personImageList
                del parsedData["batchLocationsList"]

                self.dataPackageReceiveQueue.append(parsedData)
                # self.dataPackageReceiveQueue.append((frameBytesList,
                #                                      frameCamerasList,
                #                                      frameFinedLocationsList,
                #                                      frameFeatureArrayIndexList,
                #                                      frameTimeStampList,
                #                                      processedPersonImageList,
                #                                      detectionQueueId,
                #                                      frameExtraFeatureArrayIndexList,
                #                                      personImageList,
                #                                      ))

    def dataPackageSend(self):
        while True:
            if len(self.dataPackageSendQueue) == 0:
                time.sleep(0.001)
                continue
            else:
                # frameBytesList, frameCamerasList, frameFinedLocationsList, frameFeatureArrayIndexList, frameTimeStampList, networkOutput, queueId, frameExtraFeatureArrayIndexList = self.dataPackageSendQueue.popleft()
                # self.sendGalleryPersonFeatureData(frameBytesList, frameCamerasList, frameFinedLocationsList, frameFeatureArrayIndexList, frameTimeStampList, networkOutput, queueId, frameExtraFeatureArrayIndexList)
                self.sendGalleryPersonFeatureData(self.dataPackageSendQueue.popleft())


    def fetchQueryInputData(self):
        queryRawDataBytes = self.redisConn.lpop(self.queryInputQueueName)
        if queryRawDataBytes is None:
            return None, None
        parsedData = json.loads(queryRawDataBytes)
        queryFrameBytesList = parsedData["queryImageBytesList"]
        queryFrameArrayList = [imageBytes2ImageArray(frameBytes) for frameBytes in queryFrameBytesList]
        return queryFrameArrayList, queryRawDataBytes

    def sendQueryPersonFeatureData(self, personQueryFeatureData, queryRawDataBytes):
        compressedPersonFeatureBytes, arrayMax, arrayMin = self.arrayCompress(personQueryFeatureData)
        clip = queryRawDataBytes[:-1]
        frameBytes = clip + b', "personFeatureBytes": "' + compressedPersonFeatureBytes + b'", "arrayMax": ' + bytes(str(arrayMax), encoding='utf-8') + b', "arrayMin": ' + bytes(str(arrayMin), encoding='utf-8') + b'}'
        self.redisSendConn.rpush(self.queryResultQueueName, frameBytes)

    def extractQueryFeatureData(self, queryFrameArrayList):
        return self.extractQueryListFeature(queryFrameArrayList)

    def processQueryJob(self):
        queryFrameArrayList, queryRawDataBytes = self.fetchQueryInputData()
        if queryFrameArrayList is None:
            return
        queryFeatureArray = self.extractQueryFeatureData(queryFrameArrayList)
        self.sendQueryPersonFeatureData(queryFeatureArray, queryRawDataBytes)
        # TODO：history trackDataBase 部分，

    @staticmethod
    def arrayCompress(inputArray: np.ndarray) -> tuple:
        if len(inputArray) == 0:
            return None, None, None
        arrayMax, arrayMin = np.max(inputArray), np.min(inputArray)
        normalizedArray = (((inputArray - arrayMin) / (arrayMax - arrayMin)) * 255).astype(np.uint8)
        return imageArray2ImageString(normalizedArray, quality=100), float(arrayMax), float(arrayMin)

    # def sendGalleryPersonFeatureData(self, frameBytesList,
    #                                  frameCamerasList,
    #                                  frameFinedLocationsList,
    #                                  frameFeatureArrayIndexList,
    #                                  frameTimeStampList,
    #                                  personGalleryFeatureData: np.ndarray,
    #                                  queueId,
    #                                  frameExtraFeatureArrayIndexList,
    #                                  personImageList):
    #
    #     personImageCompressedBytesList = [imageArray2ImageString(personImg) for personImg in personImageList]
    #     compressedPersonFeatureBytes, arrayMax, arrayMin = self.arrayCompress(personGalleryFeatureData)
    #     toSend = {"frameCamerasList": frameCamerasList,
    #               "frameFinedLocationsList": frameFinedLocationsList,
    #               "frameFeatureArrayIndexList": frameFeatureArrayIndexList,
    #               "frameExtraFeatureArrayIndexList": frameExtraFeatureArrayIndexList,
    #               "frameTimeStampList": frameTimeStampList,
    #               "arrayMax": arrayMax,
    #               "arrayMin": arrayMin,
    #               "frameBytesList": frameBytesList,
    #               "personImageCompressedBytesList": personImageCompressedBytesList,
    #               "personFeatureBytes": compressedPersonFeatureBytes
    #               }
    #
    #     self.redisSendConn.rpush(self.galleryResultQueueNames[queueId], json.dumps(toSend))

    def sendGalleryPersonFeatureData(self, dataPackage):

        # galleryFrameFeatureArray = self.galleryFeatureArray[arrayIndexStart:arrayIndexEnd, :]
        # galleryFrameExtraFeatureArray = self.galleryFeatureArray[extraArrayIndexStart:extraArrayIndexEnd, :]
        # galleryFramePersonImageList = self.galleryPersonImageBytesList[arrayIndexStart:arrayIndexEnd]
        # galleryFrameExtraPersonImageList = self.galleryPersonImageBytesList[extraArrayIndexStart:extraArrayIndexEnd]

        frameFeatureBytesList = []
        framePersonImageBytesList = []
        frameExtraFeatureBytesList = []
        frameExtraPersonImageBytesList = []

        personGalleryFeatureData = dataPackage["personGalleryFeatureData"]
        personImageCompressedBytesList = [imageArray2ImageString(personImg) for personImg in dataPackage["personImageList"]]

        for frameNumber, camera in enumerate(dataPackage["cameras"]):
            arrayIndexStart, arrayIndexEnd = dataPackage["frameFeatureArrayIndexList"][frameNumber]
            extraArrayIndexStart, extraArrayIndexEnd = dataPackage["frameExtraFeatureArrayIndexList"][frameNumber]
            frameFeatureBytesList.append(self.arrayCompress(personGalleryFeatureData[arrayIndexStart:arrayIndexEnd, :]) if personGalleryFeatureData is not None else (None, None, None))
            frameExtraFeatureBytesList.append(self.arrayCompress(personGalleryFeatureData[extraArrayIndexStart:extraArrayIndexEnd, :]) if personGalleryFeatureData is not None else(None, None, None))
            framePersonImageBytesList.append(personImageCompressedBytesList[arrayIndexStart:arrayIndexEnd])
            frameExtraPersonImageBytesList.append(personImageCompressedBytesList[extraArrayIndexStart:extraArrayIndexEnd])

        dataPackage["frameFeatureBytesList"] = frameFeatureBytesList
        dataPackage["frameExtraFeatureBytesList"] = frameExtraFeatureBytesList
        dataPackage["framePersonImageBytesList"] = framePersonImageBytesList
        dataPackage["frameExtraPersonImageBytesList"] = frameExtraPersonImageBytesList

        dataPackage["frameImageBytesList"] = dataPackage["frameBytesList"]
        del dataPackage["frameBytesList"]

        del dataPackage["personGalleryFeatureData"]
        del dataPackage["personImageList"]
        dataPackage["frameCamerasList"] = dataPackage["cameras"]
        del dataPackage["cameras"]
        del dataPackage["processedPersonImageList"]
        queueId = dataPackage["detectionQueueId"]
        del dataPackage["detectionQueueId"]

        # 需要特别关注的数据字段

        dataPackage["specialFields"] = ["frameFeatureBytesList",
                                             "frameExtraFeatureBytesList",
                                             "framePersonImageBytesList",
                                             "frameExtraPersonImageBytesList",
                                             "frameTimeStampList",
                                             "frameCamerasList",
                                             "frameFinedLocationsList"]
        dataPackage["featureFields"] = ["frameFeatureBytesList", "frameExtraFeatureBytesList"]
        dataPackage["imageFields"] = ["framePersonImageBytesList", "frameExtraPersonImageBytesList"]
        # toSend = {"frameCamerasList": frameCamerasList,
        #           "frameFinedLocationsList": frameFinedLocationsList,
        #           "frameFeatureArrayIndexList": frameFeatureArrayIndexList,
        #           "frameExtraFeatureArrayIndexList": frameExtraFeatureArrayIndexList,
        #           "frameTimeStampList": frameTimeStampList,
        #           "arrayMax": arrayMax,
        #           "arrayMin": arrayMin,
        #           "frameBytesList": frameBytesList,
        #           "personImageCompressedBytesList": personImageCompressedBytesList,
        #           "personFeatureBytes": compressedPersonFeatureBytes
        #           }


        self.redisSendConn.rpush(self.galleryResultQueueNames[queueId], json.dumps(dataPackage))


class OSNetFeatureExtractor(FeatureExtractorWrapper):

    def __init__(self, registerDetectionProcessIds: list, cameraIds:list, deviceId: str, batchSize: int = 32) -> None:
        super().__init__(registerDetectionProcessIds, cameraIds, deviceId, batchSize)
        self.model = osnet_ain_x1_0()
        self.model = self.model.to(self.device)
        self.model.eval()


    def preProcess(self, data):
        '''

        :param data: person image list of array
        :return:
        '''
        prePorcessedImageList = []
        for singleImage in data:
            singleImageResize = cv2.cvtColor(singleImage, cv2.COLOR_BGR2RGB)
            singleImageResize = np.divide(singleImageResize, 255.0, dtype=np.float32)
            singleImageResizeNorm = (singleImageResize - IMAGE_MEAN) / IMAGE_STD
            prePorcessedImageList.append(torch.from_numpy(np.transpose(singleImageResizeNorm, (2, 0, 1))))

        return prePorcessedImageList

    # ===================== only for query ====================
    @staticmethod
    def queryPreProcessing(img):
        """
        :param img: cv2 image, np.ndarray
        :return:
        """
        # by opencv method
        img_re = resize_and_pad(img)
        img_re = cv2.cvtColor(img_re, cv2.COLOR_BGR2RGB)
        img_re = np.divide(img_re, 255.0, dtype=np.float32)
        img_norm = (img_re - IMAGE_MEAN) / IMAGE_STD

        return np.transpose(img_norm, (2, 0, 1))

    def extractQueryBatchFeature(self, imageBatch: list) -> list:
        """
        :param imageBatch: List of images(np.ndarray)
        :return: list of features(np.ndarray 1-d)
        """
        input_batch = np.stack([self.queryPreProcessing(img) for img in imageBatch])
        input_batch = torch.from_numpy(input_batch)
        input_batch = input_batch.to(self.device)
        # with torch.no_grad():
        #     batchFeat_gpu = self.model(input_batch)
        #     batchFeat_cpu = batchFeat_gpu.cpu().numpy()
        #     return batchFeat_cpu
        with torch.no_grad():
            batchConvFeat_gpu, batchFcFeat_gpu = self.model(input_batch)
            batchConvFeat_cpu = batchConvFeat_gpu.cpu().numpy()[:, :, 7:9, 3:5]
            batchConvFeat_cpu = np.mean(batchConvFeat_cpu, axis=(2, 3))
            batchFcFeat_cpu = batchFcFeat_gpu.cpu().numpy()
            batchFeat_cpu = np.concatenate((batchConvFeat_cpu, batchFcFeat_cpu), axis=1)
            return batchFeat_cpu

    def extractQueryListFeature(self, imageList):
        if len(imageList) == 0:
            return None

        image_num = len(imageList)
        imageFeat = []
        inference_times = image_num // self.batchSize
        extra_images = imageList[inference_times * self.batchSize:]
        for i in range(1, inference_times + 1):
            start = (i - 1) * self.batchSize
            end = i * self.batchSize
            cur_imageBatch = imageList[start:end]
            batchFeat = self.extractQueryBatchFeature(cur_imageBatch)
            imageFeat.append(batchFeat)

        if extra_images:
            extra_batchFeat = self.extractQueryBatchFeature(extra_images)
            imageFeat.append(extra_batchFeat)

        return np.concatenate(imageFeat, axis=0)

    # ============================================

    def extractFeatureData(self):
        if len(self.dataPackageReceiveQueue) == 0:
            return None
        startTime = time.time()
        # frameBytesList, frameCamerasList, frameFinedLocationsList, frameFeatureArrayIndexList, frameTimeStampList, personProcessedImageList, queueId, frameExtraFeatureArrayIndexList, personImageList = self.dataPackageReceiveQueue.popleft()
        dataPackage = self.dataPackageReceiveQueue.popleft()
        dataPackage["personGalleryFeatureData"] = self.extractListFeature(dataPackage["processedPersonImageList"])
        # processedData = (frameBytesList,
        #                  frameCamerasList,
        #                  frameFinedLocationsList,
        #                  frameFeatureArrayIndexList,
        #                  frameTimeStampList,
        #                  self.extractListFeature(personProcessedImageList),
        #                  queueId,
        #                  frameExtraFeatureArrayIndexList,
        #                  personImageList,
        #                  )
        personImageListLength = len(dataPackage["personImageList"])
        print(f"Finish {personImageListLength}: {time.time() - startTime} ")
        return dataPackage

    def extractBatchFeature(self, imageBatch: list) -> list:
        """
        :param imageBatch: List of images(tensor)
        :return: list of features(np.ndarray 1-d)
        """
        input_batch = torch.stack(imageBatch)
        input_batch = input_batch.to(self.device)

        # with torch.no_grad():
        #     batchFeat_gpu = self.model(input_batch)
        #     batchFeat_cpu = batchFeat_gpu.cpu().numpy()
        #     return batchFeat_cpu
        with torch.no_grad():
            batchConvFeat_gpu, batchFcFeat_gpu = self.model(input_batch)
            batchConvFeat_cpu = batchConvFeat_gpu.cpu().numpy()[:, :, 7:9, 3:5]
            batchConvFeat_cpu = np.mean(batchConvFeat_cpu, axis=(2, 3))
            batchFcFeat_cpu = batchFcFeat_gpu.cpu().numpy()
            batchFeat_cpu = np.concatenate((batchConvFeat_cpu, batchFcFeat_cpu), axis=1)
            return batchFeat_cpu

    def extractListFeature(self, imageList):
        if len(imageList) == 0:
            return None

        image_num = len(imageList)
        imageFeat = []
        inference_times = image_num // self.batchSize
        extra_images = imageList[inference_times * self.batchSize:]
        for i in range(1, inference_times + 1):
            start = (i - 1) * self.batchSize
            end = i * self.batchSize
            cur_imageBatch = imageList[start:end]
            batchFeat = self.extractBatchFeature(cur_imageBatch)
            imageFeat.append(batchFeat)

        if extra_images:
            extra_batchFeat = self.extractBatchFeature(extra_images)
            imageFeat.append(extra_batchFeat)

        return np.concatenate(imageFeat, axis=0)


def PersonFeatureExtractorFactory(registerDetectionProcessIds, cameraIds, registerDeviceId, registerBatchSize, timeSleep=0.001):
    print("PersonFeatureExtractorFactory")
    registerFeatureExtractor = OSNetFeatureExtractor(registerDetectionProcessIds, cameraIds, registerDeviceId, registerBatchSize)
    while True:
        registerFeatureExtractor.doMyJob()
        time.sleep(timeSleep)


if __name__ == "__main__":
    print("PersonFeatureExtractorLauncher")

    # Register Cameras
    dataRootPath = 'Z:/数据/双流 ReID 测试用数据2/videos'
    # cameraIds = os.listdir(dataRootPath)
    cameraIds = list(str(item) for item in range(0, 60))

    # STAND ALONE 测试
    featureExtractor = OSNetFeatureExtractor(cameraIds, 'cuda:1')
    while True:
        featureExtractor.doMyJob()
        time.sleep(0.01)
    #
    # cameraids = ["c" + str(i % 4) for i in range(4)]
    # featureExtractor = OSNetFeatureExtractor(cameraids, "cuda:0")
    # featureExtractor.doMyJob()
    # featureExtractor.doMyJob()
    # featureExtractor.doMyJob()
    # resultQueueName = featureExtractor.galleryResultQueueName
    # tempRedis = redis.StrictRedis()
    # tempData = tempRedis.lpop(resultQueueName)
    # parsedTempData = json.loads(tempData)
    # featureBytes = parsedTempData["personFeatureBytes"]
    # featureArray = featureBytes2Array(featureBytes)
    # cv2.imwrite("./weights/featureImage.png", featureArray)

