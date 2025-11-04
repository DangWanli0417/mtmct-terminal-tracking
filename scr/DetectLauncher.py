# -*- coding:utf-8 -*-
# @Author : Re
# @Time : 2020/11/10 14:53

import numpy as np
import cv2
from abc import abstractmethod
import redis
import base64
import torch
import json
import time
import os
from math import ceil
import encodings.idna
import torch.backends.cudnn as cudnn
#
# torch.backends.cuda.matmul.allow_tf32 = True
cudnn.benchmark = False
# cudnn.deterministic = False
# cudnn.allow_tf32 = True
# torch.cuda.synchronize()


from copy import deepcopy

# backends
from utils.backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.local_detection_model_utils import invert_affine, postprocess, preprocess_video, STANDARD_COLORS, standard_to_bgr, \
    preprocess_video_crowdDet, postprocess_crowdDet, invert_affine_crowdDet, \
    preprocess_video_yolov5, postprocess_yolov5, invert_affine_yolov5


from utils.img_utils import imageArray2ImageString, imageBytes2ImageArray
from collections import deque
from threading import Thread
import random

from CrowdDet.config import config as crowdDetConfig
from CrowdDet.network import Network as crowdDetNetwork

from yolov5.YoloV5Network import getYoloV5Network

with open('./configs/DefaultConfigs.json') as fileReader:
    jsonData = json.load(fileReader)

DETECTION_REDIS_CONN_HOST = jsonData["DetectionRedisConnHost"]
DETECTION_REDIS_CONN_PORT = jsonData["DetectionRedisConnPort"]
DETECTION_REDIS_SEND_CONN_HOST = jsonData["DetectionRedisSendConnHost"]
DETECTION_REDIS_SEND_CONN_PORT = jsonData["DetectionRedisSendConnPort"]
COUNT_TASK_SWITCH = jsonData['CountTaskSwitch']
TRACK_TASK_SWITCH = jsonData['TrackTaskSwitch']

IS_SAVE_PIC = jsonData["IsSavePic"]
SAVE_PATH = jsonData["SavePath"]
RESOLUTION_SWITCH = jsonData["ResolutionSwitch"]

VIDEO_ACCESS_MAX_NUMBER = jsonData["videoAccessMaxNumber"]

DETECTOR = jsonData["Detector"]
DETECTOR_MODEL_WEIGHTS_PATH = jsonData["DetectorModelWeightsPath"]

BBOX_COLOR = (255, 0, 0)
TEXT_COLOR = (0, 255, 0)


#TODO: 将InputRegion的局部坐标映射回全局坐标


class DetectionWrapper:
    def __init__(self, registerProcessId, cameraId2Address: dict, deviceId: str, imgListSize: int, batchSize:int) -> None:
        super().__init__()
        # 参数配置
        self.countTaskSwitch = COUNT_TASK_SWITCH
        self.trackTaskSwitch = TRACK_TASK_SWITCH
        self.savePath = SAVE_PATH

        # 线程池
        self.threadPools = []
        # 计数任务的配置
        self.countMessagesQueueName = f"CountMessagesQueue"

        # 基础设置
        self.cameraIds = list(cameraId2Address.keys())
        self.registerProcessId = registerProcessId
        self.device = torch.device(deviceId)
        # self.__detectModel = detectModel
        # CNN batchSize for one forward
        self.batchSize = batchSize
        # 数据包中的图像数量
        self.imgListSize = imgListSize
        self.resultQueueName = f"Detection_{self.registerProcessId}"
        self.cameraId2FrameQueueName = {cameraId: f"Frames_{cameraId}" for cameraId in self.cameraIds}

        self.cameraId2Address = cameraId2Address

        # read camera configs
        self.cameraConfigs = {}
        for cameraId in self.cameraIds:
            if os.path.exists(f"./configs/{cameraId}.json"):
                with open(f"./configs/{cameraId}.json") as f:
                    if cameraId in self.cameraConfigs:
                        self.cameraConfigs[cameraId] = json.load(f)

        # 外部连接配置
        self.redisConn = redis.StrictRedis(host=DETECTION_REDIS_CONN_HOST, port=DETECTION_REDIS_CONN_PORT)
        self.redisSendConn = redis.StrictRedis(host=DETECTION_REDIS_SEND_CONN_HOST, port=DETECTION_REDIS_SEND_CONN_PORT)
        # self.redisConn.delete(self.resultQueueName)

        self.frameDataPool = deque(maxlen=64)
        self.cameraId2FetchThread = {cameraId: Thread(target=self.cameraDataFetch, args=(cameraId, self.cameraId2Address[cameraId])) for cameraId in self.cameraIds}
        for cameraId, cameraThread in self.cameraId2FetchThread.items():
            cameraThread: Thread = cameraThread
            cameraThread.setName(f"Fetch Thread for{cameraId}")
            self.threadPools.append(cameraThread)
            # cameraThread.start()


        # 临时数据
        self.frameBytesList = []
        self.frameArrayList = []
        self.frameTimeStampList = []

        # data receive threading
        self.dataPackageReceiveQueue = deque(maxlen=4)
        self.dataPackageReceiveEnsembleThread = Thread(target=self.dataPackageReceiveEnsemble)
        self.dataPackageReceiveEnsembleThread.setName("dataPackageReceiveEnsemble Thread")
        self.threadPools.append(self.dataPackageReceiveEnsembleThread)
        # self.dataPackageReceiveEnsembleThread.start()

        # data send threading
        self.dataPackageSendQueue = deque(maxlen=4)
        self.dataPackageSendEnsembleThread = Thread(target=self.dataPackageSend)
        self.dataPackageSendEnsembleThread.setName("dataPackageSendEnsembleThread Thread")
        self.threadPools.append(self.dataPackageSendEnsembleThread)
        # self.dataPackageSendEnsembleThread.start()

        # for to save to draw Data
        self.frameToDrawDataPool = deque()
        self.drawDetectionResultThread = Thread(target=self.drawDetectionResult)
        if IS_SAVE_PIC:
            for cameraId in self.cameraId2Address.keys():
                folderSavePath = f"{SAVE_PATH}/{cameraId}"
                os.makedirs(folderSavePath, exist_ok=True)

            self.drawDetectionResultThread.setName("drawDetectionResultThread Thread")
            self.threadPools.append(self.drawDetectionResultThread)
            # drawDetectionResultThread.start()

    def launchAllThreads(self):
        for thread in self.threadPools:
            thread.start()

    @abstractmethod
    def preProcess(self, imageArrayList):
        '''
        :param imageArrayList: 输入网络的数据
        :return:
        '''
        pass

    @abstractmethod
    def postProcess(self, data):
        '''
        :param data: 从网络输出的原始数据
        :return:
        '''
        pass

    def drawDetectionResult(self):
        while True:
            if len(self.frameToDrawDataPool) == 0:
                time.sleep(0.1)
                continue

            batchDetectedDataPackage = self.frameToDrawDataPool.popleft()
            batchDetectedFrameBytesList, batchDetectedCamerasList, batchDetectedTimeStampList, batchDetectedLocationsList = batchDetectedDataPackage["frameBytesList"], \
                                                                                            batchDetectedDataPackage["cameras"], \
                                                                                            batchDetectedDataPackage["frameTimeStampList"], \
                                                                                            batchDetectedDataPackage["batchLocationsList"]
            for frameIndex, camera in enumerate(batchDetectedCamerasList):
                frameArray = imageBytes2ImageArray(batchDetectedFrameBytesList[frameIndex])
                frameTimeStamp = batchDetectedTimeStampList[frameIndex]
                frameLocations = batchDetectedLocationsList[frameIndex]
                for personIndex, oneLocation in enumerate(frameLocations):
                    TMP_BBOX_COLOR = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.rectangle(frameArray, (oneLocation[0], oneLocation[1]), (oneLocation[2], oneLocation[3]), (0,0,255), 2)
                    #cv2.putText(frameArray, str(personIndex),  (oneLocation[0], oneLocation[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, TEXT_COLOR, 1)


                imageDstPath = f"{self.savePath}/{camera}/{str(frameTimeStamp)}.jpg"

                cv2.imwrite(imageDstPath, frameArray)


    def add2SendQueue(self, data):
        self.dataPackageSendQueue.append(data)

    def cameraDataFetch(self, cameraId, address):
        if address is None:
            redisQueueName = f"Frames_{cameraId}"
            cameraFrameCount = 0
            while True:
                cameraFrameBytes = self.redisConn.lpop(redisQueueName)

                if cameraFrameBytes is None:
                    time.sleep(0.01)
                    continue
                timeStamp = round(time.time(), ndigits=3)
                # 临时性代码：按帧数打时间戳
                #timeStamp = cameraFrameCount
                cameraFrameCount += 1
                cameraFrameString = cameraFrameBytes.decode()
                frameDatum = np.frombuffer(base64.b64decode(cameraFrameBytes), dtype="uint8")
                frameDatum = cv2.imdecode(frameDatum, cv2.IMREAD_COLOR)

                self.frameDataPool.append((cameraFrameString, frameDatum, cameraId, timeStamp))
        else:
            wrongConnectTimes = 0
            videoCap = cv2.VideoCapture(address)
            firstFrameFlag, _ = videoCap.read()
            while (not firstFrameFlag) and wrongConnectTimes < VIDEO_ACCESS_MAX_NUMBER:
                wrongConnectTimes += 1
                videoCap = cv2.VideoCapture(address)
                firstFrameFlag, _ = videoCap.read()
                time.sleep(10)

            if not firstFrameFlag:
                return

            while True:
                ret, cameraFrameArray = videoCap.read()
                if not ret:
                    time.sleep(0.6)
                    continue
                timeStamp = round(time.time(), ndigits=4)
                cameraFrameString = imageArray2ImageString(cameraFrameArray)
                self.frameDataPool.append((cameraFrameString, cameraFrameArray, cameraId, timeStamp))
                time.sleep(0.6)

    def doMyJob(self):
        toSendPackage = self.detectPersonLocations()
        if toSendPackage is not None:
            self.add2SendQueue(toSendPackage)

    def dataPackageReceiveEnsemble(self):
        while True:
            tempFrameBytesList, tempCameras, tempTimeStamp, tempFrameArrayList, tempFrameResolutionList = [], [], [], [], []
            fetchedNumber = 0
            while fetchedNumber < self.imgListSize:
                if len(self.frameDataPool) == 0:
                    time.sleep(0.01)
                    continue
                popFrameDataPool = self.frameDataPool.popleft()

                # 判断该 camera 是否配置了 config
                configFlag = False
                if popFrameDataPool[2] in self.cameraConfigs:
                    configFlag = True

                tempFrameBytesList.append(popFrameDataPool[0])

                #  若没有配置 config 则 append 原图，反之则 append 裁剪后的
                if configFlag:
                    tempROI = self.cameraConfigs[popFrameDataPool[2]]["InputRegion"]
                    tempFrameArrayList.append(popFrameDataPool[1][tempROI[0]:tempROI[2], tempROI[1]:tempROI[3]])
                    tempFrameResolutionList.append(popFrameDataPool[1].shape)
                else:
                    tempFrameArrayList.append(popFrameDataPool[1])
                    tempFrameResolutionList.append(popFrameDataPool[1].shape)

                tempCameras.append(popFrameDataPool[2])
                tempTimeStamp.append(popFrameDataPool[3])
                fetchedNumber += 1
            networkPackage = self.preProcess(tempFrameArrayList)
            # self.dataPackageReceiveQueue.append((tempFrameBytesList, tempCameras, tempTimeStamp, networkPackage))
            # self.dataPackageReceiveQueue.append((tempFrameBytesList, tempCameras, tempTimeStamp, networkPackage))
            self.dataPackageReceiveQueue.append({"cameras": tempCameras,
                                                 "frameTimeStampList": tempTimeStamp,
                                                 "frameBytesList": tempFrameBytesList,
                                                 "networkPackage": networkPackage,
                                                 "frameResolutionList": tempFrameResolutionList})

    def dataPackageSend(self):
        while True:
            if len(self.dataPackageSendQueue) == 0:
                time.sleep(0.001)
                continue
            else:
                dataPackage = self.dataPackageSendQueue.popleft()
                # tempFrameBytesList, tempCameras, tempTimeStamp, networkOutput = self.dataPackageSendQueue.popleft()
                detectedLocationsList = self.postProcess(dataPackage["networkOutput"])
                dataPackage["batchLocationsList"] = detectedLocationsList
                del dataPackage["networkOutput"]
                if not RESOLUTION_SWITCH:
                    del dataPackage["frameResolutionList"]
                if IS_SAVE_PIC:
                    toDrawDataPackage = deepcopy(dataPackage)
                    self.frameToDrawDataPool.append(toDrawDataPackage)
                self.sendLocations(dataPackage)

    @abstractmethod
    def detectPersonLocations(self) -> list:
        """
        注意：
        1. 针对不同的检测模型，对__frames需要完成各自的预处理
        2. 必须在return之前，调用sendLocations方法

        :return: list of [x0, y0, x1, y1]
        """
        pass

    def bytesDecode2Array(self):
        tempFrameArrayList = []
        for frameBytes in self.frameBytesList:
            frameDatum = np.frombuffer(base64.b64decode(frameBytes), dtype="uint8")
            # frameDatum = np.frombuffer(frameBytes, dtype="uint8")
            frameDatum = cv2.imdecode(frameDatum, cv2.IMREAD_COLOR)
            tempFrameArrayList.append(frameDatum)
        self.frameArrayList = tempFrameArrayList

    def fetchCameraFrameBytes(self, cameraId):
        return self.redisConn.lpop(self.cameraId2FrameQueueName[cameraId])

    def sendLocations(self, dataPackage: dict):
    # def sendLocations(self, frameLocationsList, frameCameraList, frameTimeStampList, frameBytesList):
        """
        #TODO: 改为线程执行模式
        :return:
        """
        if self.trackTaskSwitch:
            toSend = json.dumps(dataPackage)
            self.redisSendConn.rpush(self.resultQueueName, toSend)
        if self.countTaskSwitch:
            del dataPackage["frameBytesList"]
            toSend = json.dumps(dataPackage)
            self.redisSendConn.rpush(self.countMessagesQueueName, toSend)


class EfficientModelWrapper(DetectionWrapper):

    def __init__(self, registerProcessId, cameraId2Address: dict, deviceId: str, imgListSize: int, batchSize: int) -> None:
        super().__init__(registerProcessId, cameraId2Address, deviceId, imgListSize, batchSize)

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.color_list = standard_to_bgr(STANDARD_COLORS)

        force_input_size = None  # set None to use default size
        self.threshold = 0.2
        self.iou_threshold = 0.2
        self.compound_coef = 3
        self.obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

        # tf bilinear interpolation is different from any other's, just make do
        force_input_size = None
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.input_size = input_sizes[self.compound_coef] if force_input_size is None else force_input_size
        self.model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(self.obj_list))
        self.model.load_state_dict(torch.load(DETECTOR_MODEL_WEIGHTS_PATH))
        self.model.half()
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(self.device)
        self.launchAllThreads()

    def postProcess(self, data):
        framed_metas, imgListOutResult, len_ori_imgs = data
        imgListOut = invert_affine(framed_metas, imgListOutResult)
        _detectedPedestrianLocations = self.postProcessing(imgListOut, len_ori_imgs)

        return _detectedPedestrianLocations

    def preProcess(self, imageArrayList):
        return preprocess_video(imageArrayList, self.input_size)


    @staticmethod
    def postProcessing(preds, frameNums):
        personsLocationsList = []

        for i in range(frameNums):
            # if len(preds[i]['rois']) == 0:
            #     continue

            singlePersonLocations = []
            for j in range(len(preds[i]['rois'])):
                if preds[i]['class_ids'][j] == 0:
                    # x1, y1, x2, y2 = (int(aInt) for aInt in preds[i]['rois'][j].astype(np.int))
                    x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int).tolist()
                    singlePersonLocations.append((int(x1), int(y1), int(x2), int(y2)))
            personsLocationsList.append(singlePersonLocations)

        return personsLocationsList

    def detectPersonLocations(self):
        startTime = time.time()
        if len(self.dataPackageReceiveQueue) == 0:
            return
        dataPackage = self.dataPackageReceiveQueue.popleft()
        ori_imgs, framed_imgs, framed_metas = dataPackage["networkPackage"]
        imgListOutResult = []
        for forwardTime in range(ceil(len(framed_imgs) / self.batchSize)):
            start = forwardTime * self.batchSize
            end = min((forwardTime + 1) * self.batchSize, self.imgListSize)
            batchImages = framed_imgs[start:end]
            x = torch.stack([torch.from_numpy(fi) for fi in batchImages], 0).to(self.device)
            x = x.permute(0, 3, 1, 2)
            with torch.no_grad():
                features, regression, classification, anchors = self.model(x)
                out = postprocess(x,
                                  anchors, regression, classification,
                                  self.regressBoxes, self.clipBoxes,
                                  self.threshold, self.iou_threshold)
                imgListOutResult.extend(out)
        endTime = time.time() - startTime
        print(f"Finish {len(ori_imgs)} images: {endTime}, framePoolLength: {len(self.frameDataPool)}, dataPackagesLength: {len(self.dataPackageReceiveQueue)}")
        dataPackage["networkOutput"] = (framed_metas, imgListOutResult, len(ori_imgs))
        del dataPackage["networkPackage"]
        return dataPackage


class CrowdDetModelWrapper(DetectionWrapper):
    def __init__(self, registerProcessId, cameraId2Address: dict, deviceId: str, imgListSize: int,
                 batchSize: int) -> None:
        super().__init__(registerProcessId, cameraId2Address, deviceId, imgListSize, batchSize)

        self.color_list = standard_to_bgr(STANDARD_COLORS)

        self.modelConfig = crowdDetConfig

        self.model = crowdDetNetwork()
        checkpoint = torch.load(DETECTOR_MODEL_WEIGHTS_PATH)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.half()
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(self.device)
        self.launchAllThreads()

    def postProcess(self, data):
        framed_metas, imgListOutResult, len_ori_imgs = data
        imgListOut = invert_affine_crowdDet(framed_metas, imgListOutResult, self.modelConfig.eval_image_max_size, self.modelConfig.eval_image_short_size)
        _detectedPedestrianLocations = self.postProcessing(imgListOut, len_ori_imgs)

        return _detectedPedestrianLocations

    def preProcess(self, imageArrayList):
        return preprocess_video_crowdDet(imageArrayList,
                                         self.modelConfig.eval_image_max_size, self.modelConfig.eval_image_short_size,
                                         self.modelConfig.image_mean, self.modelConfig.image_std)



    @staticmethod
    def postProcessing(preds, frameNums):
        personsLocationsList = []

        for i in range(frameNums):
            singlePersonLocations = []
            for j in range(len(preds[i]['rois'])):
                if int(preds[i]['class_ids'][j]) == 1:
                    # x1, y1, x2, y2 = (int(aInt) for aInt in preds[i]['rois'][j].astype(np.int))
                    x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int).tolist()
                    singlePersonLocations.append((int(x1), int(y1), int(x2), int(y2)))
            personsLocationsList.append(singlePersonLocations)

        return personsLocationsList

    def detectPersonLocations(self):
        startTime = time.time()
        if len(self.dataPackageReceiveQueue) == 0:
            return
        dataPackage = self.dataPackageReceiveQueue.popleft()
        ori_imgs, framed_imgs, framed_metas = dataPackage["networkPackage"]
        imgListOutResult = []
        for forwardTime in range(ceil(len(framed_imgs) / self.batchSize)):
            start = forwardTime * self.batchSize
            end = min((forwardTime + 1) * self.batchSize, self.imgListSize)
            batchImages = framed_imgs[start:end]
            x = torch.stack([torch.from_numpy(fi) for fi in batchImages], 0).to(self.device)
            x = x.permute(0, 3, 1, 2)
            with torch.no_grad():
                pred_boxes_batch = self.model(x)
                out = postprocess_crowdDet(pred_boxes_batch,
                                           self.modelConfig.pred_cls_threshold,
                                           self.modelConfig.test_nms_method,
                                           self.modelConfig.test_nms,
                                           self.modelConfig.visulize_threshold
                                           )
                imgListOutResult.extend(out)
        endTime = time.time() - startTime
        print(
            f"Finish {len(ori_imgs)} images: {endTime}, framePoolLength: {len(self.frameDataPool)}, dataPackagesLength: {len(self.dataPackageReceiveQueue)}")
        dataPackage["networkOutput"] = (framed_metas, imgListOutResult, len(ori_imgs))
        del dataPackage["networkPackage"]
        return dataPackage


class YoloV5ModelWrapper(DetectionWrapper):
    def __init__(self, registerProcessId, cameraId2Address: dict, deviceId: str, imgListSize: int, batchSize: int) -> None:
        super().__init__(registerProcessId, cameraId2Address, deviceId, imgListSize, batchSize)


        self.threshold = 0.3
        self.iou_threshold = 0.45
        self.obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']


        self.stride = 64
        self.input_size = 640
        self.firstFlag = True
        # self.model = attempt_load(DETECTOR_MODEL_WEIGHTS_PATH, map_location=self.device)
        #
        #
        # self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        #


        self.model = getYoloV5Network(DETECTOR_MODEL_WEIGHTS_PATH, self.device)
        #self.model.to(self.device)

        self.launchAllThreads()

    def postProcess(self, data):
        framed_metas, imgListOutResult, len_ori_imgs = data
        imgListOut = invert_affine_yolov5(framed_metas, imgListOutResult, self.input_size, self.input_size)
        _detectedPedestrianLocations = self.postProcessing(imgListOut, len_ori_imgs)

        return _detectedPedestrianLocations

    def preProcess(self, imageArrayList):
        return preprocess_video_yolov5(imageArrayList, self.input_size, stride=self.stride)


    @staticmethod
    def postProcessing(preds, frameNums):
        personsLocationsList = []

        for i in range(frameNums):
            # if len(preds[i]['rois']) == 0:
            #     continue

            singlePersonLocations = []
            for j in range(len(preds[i]['rois'])):
                #if preds[i]['class_ids'][j] == 2 or preds[i]["class_ids"][j] == 5 or preds[i]["class_ids"][j] == 7:
                if preds[i]['class_ids'][j] == 0:
                    # x1, y1, x2, y2 = (int(aInt) for aInt in preds[i]['rois'][j].astype(np.int))
                    x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int).tolist()
                    singlePersonLocations.append((int(x1), int(y1), int(x2), int(y2)))
            personsLocationsList.append(singlePersonLocations)

        return personsLocationsList

    @torch.no_grad()
    def detectPersonLocations(self):
        startTime = time.time()
        if len(self.dataPackageReceiveQueue) == 0:
            return
        dataPackage = self.dataPackageReceiveQueue.popleft()
        ori_imgs, framed_imgs, framed_metas = dataPackage["networkPackage"]
        imgListOutResult = []
        for forwardTime in range(ceil(len(framed_imgs) / self.batchSize)):
            start = forwardTime * self.batchSize
            end = min((forwardTime + 1) * self.batchSize, self.imgListSize)
            batchImages = framed_imgs[start:end]
            x = torch.stack([torch.from_numpy(fi) for fi in batchImages], 0)
            x = x.permute(0, 3, 1, 2).to(self.device)
            if self.firstFlag:
                x = np.load('./imgstack.npy')
                x = torch.from_numpy(x).to(self.device)
                self.firstFlag = False
        #
        # with torch.no_grad():
            print(x.get_device())
            pred_boxes_batch = self.model(x, augment=False, visualize=False)[0]
            out = postprocess_yolov5(pred_boxes_batch, self.threshold, self.iou_threshold)
            imgListOutResult.extend(out)

        endTime = time.time() - startTime
        print(f"Finish {len(ori_imgs)} images: {endTime}, framePoolLength: {len(self.frameDataPool)}, dataPackagesLength: {len(self.dataPackageReceiveQueue)}")
        dataPackage["networkOutput"] = (framed_metas, imgListOutResult, len(ori_imgs))
        del dataPackage["networkPackage"]
        return dataPackage


def DetectionFactory(registerProcessId, registerCameraId2Address, registerDeviceId, registerListSize, registerBatchSize, timeSleep=0.001):
    if DETECTOR == "EfficientDet":
        print("DetectionFactory: EfficientDet")
        registerDetector = EfficientModelWrapper(registerProcessId, registerCameraId2Address, registerDeviceId,
                                                 registerListSize, registerBatchSize)
    elif DETECTOR == "CrowdDet":
        print("DetectionFactory: CrowdDet")
        registerDetector = CrowdDetModelWrapper(registerProcessId, registerCameraId2Address, registerDeviceId,
                                             registerListSize, registerBatchSize)
    elif DETECTOR == "YoloV5":
        print("DetectionFactory: YoloV5")
        registerDetector = YoloV5ModelWrapper(registerProcessId, registerCameraId2Address, registerDeviceId,
                                             registerListSize, registerBatchSize)
    else:
        raise ValueError("No such Detector!")
    while True:
        registerDetector.doMyJob()
        time.sleep(timeSleep)


if __name__ == "__main__":
    # Register Cameras

    cameraId2Address = {1: "rtsp://10.10.10.111/0", 2: "rtsp://10.10.10.111/1"}

    # STAND ALONE 测试
    efficientModel = DetectionFactory(1, cameraId2Address, 'cuda:0', 24, 8)
    while True:
        efficientModel.doMyJob()

    print("done")






