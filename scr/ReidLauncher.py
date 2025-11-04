# -*- coding: utf-8 -*-
# @Time: 2020/11/10 13:48
# @Author: libenchao
# @Contact: libenchao@caacsri.com
# @FileName: ReidLauncher.py
import numpy as np
import cv2
import math
import os
import time
from collections import defaultdict, deque
from utils.reid_utils import *
from utils.img_utils import *
import redis
import json
from threading import Thread
import encodings.idna

from AttributeModule import Res50AttributeExtractor, DummyAttributeExtractor

# TODO: use PIL read image test the procedure...

with open('./configs/DefaultConfigs.json') as fileReader:
    jsonData = json.load(fileReader)
    
REDIS_CONN_HOST = jsonData["ReIDRedisConnHost"]
REDIS_CONN_PORT = jsonData["ReIDRedisConnPort"]
REDIS_SEND_CONN_HOST = jsonData["ReIDRedisSendConnHost"]
REDIS_SEND_CONN_PORT = jsonData["ReIDRedisSendConnPort"]
REDIS_TRACK_CONN_HOST = jsonData["ReIDTrackRedisConnHost"]
REDIS_TRACK_CONN_PORT = jsonData["ReIDTrackRedisConnPort"]

TRACK_DATABASE_FLAG = jsonData["TrackDataBaseFlag"]

# USEFUL_ATTRIBUTES = ["LowerLength", "LowerType", "UpperColor", "LowerColor"]
USEFUL_ATTRIBUTES = ["LowerLength", "UpperColor", "LowerColor"]
ATTRIBUTE_NAME_TO_CLASS = {"Res50": Res50AttributeExtractor, "Dummy": DummyAttributeExtractor}


class ReIdTrack:
    def __init__(self, cameraId, initialId, initialTimeStamp, initialFeature, initialExtraFeature, initialLocation, initialImageArray, initialImage, initialExtraImage) -> None:
        super().__init__()

        self.cameraId = cameraId
        self.id = initialId
        self.initialLocation = initialLocation
        self.initialTimestamp = initialTimeStamp
        self.lastLocation = initialLocation
        self.lastCenter = (int((self.lastLocation[0] + self.lastLocation[2]) / 2), int((self.lastLocation[1] + self.lastLocation[3]) / 2))
        self.lastTimestamp = initialTimeStamp
        self.lastFeature = initialFeature
        # 局部person的图像
        self.lastImageArray = initialImageArray
        self.appearTime = 0
        self.idCounter = {}
        self.scoreMap = {}
        self.disappearTime = 0
        self.betaSimilarity = 0.
        self.lastPearson = 0.
        self.historyLocations = [initialLocation]
        self.historyTimestamp = [initialTimeStamp]
        self.historyFeature = [initialFeature]
        self.historyImage = [initialImage]

        if initialExtraFeature is not None:
            self.historyExtraFeature = [initialExtraFeature]
            self.historyExtraImage = [initialExtraImage]
        else:
            self.historyExtraFeature = None
            self.historyExtraImage = None
        self.featureSampleSize = 3
        self.tag = None
        self.bytesNameList = ("imageBytesList", "extraImageBytesList", "personFeatureBytes")

    def __str__(self):
        return str(self.id)

    def labelTag(self, newTags):
        self.tag = newTags

    def update(self, newLocation, newFeature, newExtraFeature, timeStampSecond, newImage, newExtraImage):
        lastWidth = self.lastLocation[2] - self.lastLocation[0]
        newCenter = (int((newLocation[0] + newLocation[2]) / 2), int((newLocation[1] + newLocation[3]) / 2))
        displacement = math.sqrt(math.pow(self.lastCenter[0] - newCenter[0], 2) + math.pow(self.lastCenter[1] - newCenter[1], 2))
        normalizedDisplacement = displacement / lastWidth / (timeStampSecond - self.lastTimestamp)
        if normalizedDisplacement < 2:
            self.lastLocation, self.lastCenter, self.lastFeature = newLocation, newCenter, newFeature
            self.appearTime += 1
            self.disappearTime = 0
            self.lastTimestamp = timeStampSecond
            self.historyFeature.append(newFeature)
            self.historyImage.append(newImage)
            if newExtraFeature is not None:
                self.historyExtraFeature.append(newExtraFeature)
                self.historyExtraImage.append(newExtraImage)
            self.historyLocations.append(newLocation)
            self.historyTimestamp.append(timeStampSecond)
            return True
        else:
            self.disappearTime += 1
            return False

    def delQueryPersonIds(self, toDeleteQueryPersonIds: list):
        for toDeleteQueryPersonId in toDeleteQueryPersonIds:
            del self.idCounter[toDeleteQueryPersonId]
            del self.scoreMap[toDeleteQueryPersonId]

    def addMatchedId(self, reidMatchedQueryIds, reidMatchedQueryTags, reidMatchedSimilarities):
        if self.tag is None:
            return
        else:
            for enumId, reidMatchedId in enumerate(reidMatchedQueryIds):
                reidPersonTag = reidMatchedQueryTags[enumId]
                tagChecks = [checkedAttribute not in self.tag or self.tag[checkedAttribute] is None or reidPersonTag[checkedAttribute] is None or (reidPersonTag[checkedAttribute] == self.tag[checkedAttribute]) for checkedAttribute in USEFUL_ATTRIBUTES]
                if all(tagChecks):
                    self.idCounter[reidMatchedId] = self.idCounter.get(reidMatchedId, 0) + 1
                    self.scoreMap[reidMatchedId] = reidMatchedSimilarities[enumId]
        # self.idCounter.update(matchedIds)

    @property
    def confirmedId(self):
        # 筛选出匹配了1次（含）以上的query对应的id
        validatedQueryPersonIds = [queryPersonId for queryPersonId, queryPersonReIdMatchedTime in self.idCounter.items() if queryPersonReIdMatchedTime >= 1]
        if len(validatedQueryPersonIds) == 0:
            return None
        else:
            return validatedQueryPersonIds

        # 写图逻辑

    @property
    def checkIntegrity(self):
        return len(self.historyFeature) >= self.featureSampleSize

    # data sent to HistoryTrackDatabase
    @property
    def datum2send(self) -> bytes:
        if self.historyExtraFeature is not None:
            featureBytes2Send = imageArray2ImageString(np.concatenate((np.stack(self.historyFeature), np.stack(self.historyExtraFeature))))
        else:
            featureBytes2Send = imageArray2ImageString(np.stack(self.historyFeature))
        # 在checkIntegrity过后，能保证该轨迹至少已经进行过2次跟踪，因此必有tag
        toSendString = json.dumps({"trackLocations": self.historyLocations,
                                   "trackTimeStamp": self.historyTimestamp,
                                   "tags": self.tag,
                                   "cameraId": self.id,
                                   "imageBytesList": self.historyImage,
                                   "extraImageBytesList": self.historyExtraImage,
                                   "featureBytes2Send": featureBytes2Send,
                                   })
        return toSendString


class Camera:

    def __init__(self, cameraId, reidTaskSwitch=True, trackTaskSwitch=True) -> None:
        super().__init__()
        self.id = cameraId
        self.cameraTracker = CameraTracker(cameraId)
        self.cameraReIdFinder = ReIdTrackMatcher(cameraId)

        self.reidTaskSwitch = reidTaskSwitch
        self.trackTaskSwitch = trackTaskSwitch
        os.makedirs(f"../reidResult/{self.id}/", exist_ok=True)
        print(f"Camera {self.id} folder created.")

    def __str__(self) -> str:
        return self.id

    @property
    def totalTracksNumber(self):
        return len(self.cameraTracker.lastTrackPool)

    def drawResult(self, rawImageArray, timePoint, unMatchedTracks, rois, queryImagesList, queryIds: list):
        # draw result
        annotatedImages = drawBbox(rois, rawImageArray)
        everReIdMatched = False
        everMatchedReIdPersonIds = set()
        for nowTrack in self.cameraTracker.lastTrackPool:
            if nowTrack not in unMatchedTracks:
                coord = nowTrack.lastLocation
                c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
                confirmedInfo = nowTrack.confirmedId
                if confirmedInfo is not None:
                    everReIdMatched = True
                    everMatchedReIdPersonIds.update(confirmedInfo)
                    cv2.putText(annotatedImages,
                                f"{nowTrack.id}, reidBeta:{confirmedInfo}",
                                (c1[0], c1[1] - 2), 0, 0.5,
                                [255, 255, 255],
                                thickness=1, lineType=cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(annotatedImages, f"{nowTrack.id}",
                                (c1[0], c1[1] - 2), 0, 0.5,
                                [0, 0, 0],
                                thickness=1, lineType=cv2.FONT_HERSHEY_SIMPLEX)
        displayPictureHeight, displayPictureWidth, displayCharHeight = 256, 128, 12
        displayRowNumber, displayColumnNumber = 0, 0
        for everMatchedReIdPictureId in everMatchedReIdPersonIds:
            if displayColumnNumber >= 8:
                displayRowNumber += 1
                displayColumnNumber = 0
            displayAreaHeightStart = (displayCharHeight + displayCharHeight + displayPictureHeight) * displayRowNumber
            displayAreaWidthStart = displayPictureWidth * displayColumnNumber
            matchedReIdImage = queryImagesList[queryIds.index(everMatchedReIdPictureId)]
            annotatedImages[displayAreaHeightStart:displayAreaHeightStart + displayPictureHeight, displayAreaWidthStart:displayAreaWidthStart + displayPictureWidth, :] = matchedReIdImage
            cv2.putText(annotatedImages, f"{everMatchedReIdPictureId}",
                        (displayAreaWidthStart + 64, displayAreaHeightStart + displayPictureHeight), 0, 0.5,
                        [255, 255, 255],
                        thickness=1, lineType=cv2.FONT_HERSHEY_SIMPLEX)
            displayColumnNumber += 1

        # save image, only save those reid matched frames.
        if everReIdMatched:
            print(f'{self.id}: matched at {timePoint}.')
            cv2.imwrite(f'../reidResult/{self.id}/matched_{timePoint}.jpg', annotatedImages)

    @property
    def unlabeledTracks(self):
        unlabeledTracks = [track for track in self.cameraTracker.lastTrackPool if track.tag is None]
        return unlabeledTracks

    @property
    def trackEnumeratedLastFeatureArray(self):
        tracks, lastTrackArray = [], []
        for track in self.cameraTracker.lastTrackPool:
            lastTrackArray.append(track.lastFeature)
            tracks.append(track)
        if len(tracks) > 0:
            return np.stack(lastTrackArray), tracks
        else:
            return lastTrackArray, tracks

    def updateQuery(self, newQueryFeatureList):
        pass

    def doPackageJob(self, singleFrameDataPackage, queryFeatureArray=None, queryImageArrayList=None, queryTagsList=None, queryIds=None):
        """

        :param dataPackage: dict, {frameTimeStampList, frameBytesList, frameExtraFeatureA}
        :param queryFeatureArray:
        :param queryImageArrayList:
        :param queryTagsList:
        :param queryIds:
        :return:
        """
        galleryFrameArray = imageBytes2ImageArray(singleFrameDataPackage["ImageBytes"]) if len(singleFrameDataPackage["FinedLocations"]) > 0 else None
        self.cameraTracker.trackingByPackage(singleFrameDataPackage, galleryFrameArray, self.trackTaskSwitch) #tracking
        if self.reidTaskSwitch and queryFeatureArray is not None and singleFrameDataPackage["ImageBytes"] is not None:
            self.cameraReIdFinder.matchingTrack(queryFeatureArray, queryTagsList, queryIds, self.trackEnumeratedLastFeatureArray)
            # self.drawResult(singleFrameDataPackage, self.cameraTracker.unMatchedTracks, queryImageArrayList, queryIds)


class CameraTracker:

    def __init__(self, cameraId) -> None:
        super().__init__()
        self.id = cameraId
        self.lastTrackPool = set()
        self.trackIdDeque = deque(range(0, 400), maxlen=400)
        self.redisConn = redis.StrictRedis()
        self.redisConnTrack = redis.StrictRedis()
        self.trackDataBaseQueueName = f"SecurityHistoryTrackDataBase"

        # tracking data send threading
        self.trackingDataPackageSendQueue = deque()

        # self.redisConn.delete(self.trackDataBaseQueueName)

        self.unMatchedTracks = []

    def __str__(self) -> str:
        return self.id

    @property
    def trackEnumeratedLastFeatureArray(self):
        tracks, lastTrackArray = [], []
        for track in self.lastTrackPool:
            lastTrackArray.append(track.lastFeature)
            tracks.append(track)
        if len(tracks) > 0:
            return np.stack(lastTrackArray), tracks
        else:
            return lastTrackArray, tracks

    @property
    def reidEverMatchedTracks(self):
        everMatchedTracks = []
        for track in self.lastTrackPool:
            confirmedInfo = track.confirmedId
            if confirmedInfo is not None:
                everMatchedTracks.append(track)
        return everMatchedTracks

    def send2HistoryTrackDataBase(self, toSendDatum):
        self.redisConn.rpush(self.trackDataBaseQueueName, toSendDatum)

    def trackingByPackage(self, singleDataPackage, galleryFrameArray, trackTaskSwitch):
        # 如果上一帧为止，轨迹池为空，则当前帧所有检测到每一个人都生成一条新的轨迹
        # 上一帧轨迹池不为空，则需要将当前帧检测到的每一个人与历史轨迹池进行对比，从而实现跟踪
        # 若当前帧没有检测到人，则需要将轨迹全部 disapeartime + 1
        # self.clearCache()

        self.unMatchedTracks = []
        usedCurrentIds = set()
        newAddedTrackPool = set()
        activeTrackPool = set()
        deadTrackPool = set()

        extraFlag = singleDataPackage["ExtraFeatureArray"] is not None
        galleryPedestrianLocations = singleDataPackage["FinedLocations"]
        timePoint = singleDataPackage["TimeStamp"]
        resolution = singleDataPackage["Resolution"]
        galleryPedestrianFeatureArray = singleDataPackage["FeatureArray"]
        galleryFramePersonBytesList = singleDataPackage["PersonImageBytes"]
        galleryPedestrianExtraFeatureArray = singleDataPackage["ExtraFeatureArray"]
        galleryFrameExtraPersonBytesList = singleDataPackage["ExtraPersonImageBytes"]

        if len(galleryPedestrianLocations) > 0:
            frameImageArray = galleryFrameArray
            if len(self.lastTrackPool) == 0:
                for galleryPedestrianId, galleryPedestrianLocation in enumerate(galleryPedestrianLocations):
                    x1, y1, x2, y2 = galleryPedestrianLocation
                    newReIdTrack = ReIdTrack(self.id,
                                             self.trackIdDeque.popleft(),
                                             timePoint,
                                             galleryPedestrianFeatureArray[galleryPedestrianId],
                                             galleryPedestrianExtraFeatureArray[galleryPedestrianId] if extraFlag else None,
                                             galleryPedestrianLocations[galleryPedestrianId],
                                             frameImageArray[y1:y2, x1:x2, :],
                                             galleryFramePersonBytesList[galleryPedestrianId],
                                             galleryFrameExtraPersonBytesList[galleryPedestrianId] if extraFlag else None)
                    self.lastTrackPool.add(newReIdTrack)
                    newAddedTrackPool.add(newReIdTrack)

                if trackTaskSwitch:
                    self.trackingDataPackageSendQueue.append((timePoint, deadTrackPool, self.lastTrackPool, self.id, resolution))
                return self.lastTrackPool

            lastTrackArray, tracks = self.trackEnumeratedLastFeatureArray
            trackArray, trackStd, trackMean, trackBias = getQueryArrayAndStdAndMeanAndBias(lastTrackArray)
            galleryArray, galleryStd, galleryMean, galleryBias = getGalleryArrayAndStdAndMeanAndBias(
                galleryPedestrianFeatureArray)
            pearsonMatrix = getPearsonArrayTrack(trackStd, trackBias, galleryStd, galleryBias)
            bestMatchedIndex = pearsonMatrix.argmax(axis=1)

            for lastTrackId, currentId in enumerate(bestMatchedIndex):
                pearSimilarity = pearsonMatrix[lastTrackId, currentId]
                track: ReIdTrack = tracks[lastTrackId]
                if pearSimilarity >= 0.65 and currentId not in usedCurrentIds:
                    usedCurrentIds.add(currentId)
                    # track.betaSimilarity = round(float(getBetaSimilarity(reidFeatures[currentId], track.lastFeature)), 4)
                    track.lastPearson = round(float(pearSimilarity), 4)
                    # print(track.betaSimilarity)
                    if track.update(galleryPedestrianLocations[currentId], galleryPedestrianFeatureArray[currentId],
                                    galleryPedestrianExtraFeatureArray[currentId] if extraFlag else None, timePoint,
                                    galleryFramePersonBytesList[currentId],
                                    galleryFrameExtraPersonBytesList[currentId] if extraFlag else None):
                        # 确定跟踪上了
                        activeTrackPool.add(track)
                    else:
                        self.unMatchedTracks.append(track)
                else:
                    track.disappearTime += 1
                    self.unMatchedTracks.append(track)

            unusedCurrentIds = set(range(0, len(galleryPedestrianLocations))).difference(usedCurrentIds)

            for unusedCurrentId in unusedCurrentIds:
                x1, y1, x2, y2 = galleryPedestrianLocations[unusedCurrentId]
                newReIdTrack = ReIdTrack(self.id,
                                         self.trackIdDeque.popleft(),
                                         timePoint,
                                         galleryPedestrianFeatureArray[unusedCurrentId],
                                         galleryPedestrianExtraFeatureArray[unusedCurrentId] if extraFlag else None,
                                         galleryPedestrianLocations[unusedCurrentId],
                                         frameImageArray[y1:y2, x1:x2, :],
                                         galleryFramePersonBytesList[unusedCurrentId],
                                         galleryFrameExtraPersonBytesList[unusedCurrentId] if extraFlag else None)
                self.lastTrackPool.add(newReIdTrack)
                newAddedTrackPool.add(newReIdTrack)
                activeTrackPool.add(newReIdTrack)
        else:
            unMatchedTracks = self.trackEnumeratedLastFeatureArray[1]
            for unMatchedTrack in unMatchedTracks:
                unMatchedTrack.disappearTime += 1

        for unMatchedTrack in self.unMatchedTracks:
            if unMatchedTrack.disappearTime >= 3:
                self.trackIdDeque.append(unMatchedTrack.id)
                deadTrackPool.add(unMatchedTrack.id)
                if TRACK_DATABASE_FLAG and unMatchedTrack.checkIntegrity:
                    self.send2HistoryTrackDataBase(unMatchedTrack.datum2send)
                self.lastTrackPool.remove(unMatchedTrack)

        if trackTaskSwitch:
            self.trackingDataPackageSendQueue.append((timePoint, deadTrackPool, self.lastTrackPool, self.id, resolution))

        return self.lastTrackPool


class ReIdTrackMatcher(ReIdMatcher):

    # reidHigh: 0.46, reidLow: 0.08
    def __init__(self, cameraId, reidHigh=0.4, reidLow=0.12) -> None:
        super().__init__(reidHigh=reidHigh, reidLow=reidLow)
        self.redisConn = redis.StrictRedis()
        self.queryReIdResultQueueName = "RealtimeQueryMatchResult"
        self.id = cameraId

        # self.redisConn.delete(self.queryReIdResultQueueName)

    def __str__(self) -> str:
        return self.id

    def send2QueryCenter(self, toSendDatum):
        self.redisConn.rpush(self.queryReIdResultQueueName, toSendDatum)

    def matchingTrack(self, inputQueryArray, queryTagsList, queryIds, trackEnumeratedLastFeatureArray):
        lastTrackArray, tracks = trackEnumeratedLastFeatureArray
        if len(tracks) > 0:
            lastTrackIndex2ReidMatchedPersonIndexList, lastTrackIndex2ReidMatchedScoresList = self.matching(inputQueryArray, lastTrackArray)
            for lastTrackIndex, reidMatchedPersonIndex in lastTrackIndex2ReidMatchedPersonIndexList.items():
                lastTrack: ReIdTrack = tracks[lastTrackIndex]
                reidMatchedPersonIds = [queryIds[reidMatchedPersonNo] for reidMatchedPersonNo in reidMatchedPersonIndex]
                reidMatchedPersonTags = [queryTagsList[reidMatchedPersonNo] for reidMatchedPersonNo in reidMatchedPersonIndex]
                lastTrack.addMatchedId(reidMatchedPersonIds, reidMatchedPersonTags, lastTrackIndex2ReidMatchedScoresList[lastTrackIndex])
                if lastTrack.checkIntegrity:
                    toSendDatum = {"queryPersonIds": reidMatchedPersonIds,
                                   "cameraId2Results": {self.id: [(lastTrack.initialTimestamp,
                                                                  lastTrack.lastTimestamp,
                                                                  lastTrack.initialLocation,
                                                                  lastTrack.lastLocation), ]}}
                    self.send2QueryCenter(json.dumps(toSendDatum))


class RealTimeProcessor:
    def __init__(self, registerDetectionProcessIds, queryFeatureResultQueueName, queryFeatureAndTagsResultQueueName, attributeLabeler) -> None:
        super().__init__()
        # 这些项需要通过配置文件来解析
        self.reidTaskSwitch = True
        self.trackTaskSwitch = True
        self.attributeTaskSwitch = False
        self.colorFilterSwitch = False

        # 基础数据定义
        # self.cameraIds = cameraIds
        # self.cameraPool = {cameraId: Camera(cameraId, self.reidTaskSwitch, self.trackTaskSwitch) for cameraId in cameraIds}
        self.cameraPool = {}
        self.redisConn = redis.StrictRedis(host=REDIS_CONN_HOST, port=REDIS_CONN_PORT)
        self.redisSendConn = redis.StrictRedis(host=REDIS_SEND_CONN_HOST, port=REDIS_SEND_CONN_PORT)
        # self.galleryFeatureResultQueueName = galleryFeatureResultQueueName
        # self.galleryFeatureResultQueueNumber = galleryFeatureResultQueueNumber
        self.registerDetectionProcessIds = registerDetectionProcessIds
        self.galleryFeatureResultQueueNames = [f"Gallery_ReidFeature_{qId}" for qId in self.registerDetectionProcessIds]
        self.queryFeatureResultQueueName = queryFeatureResultQueueName
        self.queryFeatureAndTagsResultQueueName = queryFeatureAndTagsResultQueueName

        # 临时数据定义
        self.galleryFrameCamerasList = []
        self.galleryFrameFinedLocationsList = []
        self.galleryFrameFeatureArrayIndexList = []
        self.galleryFrameExtraFeatureArrayIndexList = []
        self.galleryFrameTimeStampList = []
        self.galleryFeatureArray = None
        self.galleryFrameBytesList = []
        self.galleryColorArray = None
        self.galleryPersonImageBytesList = None

        self.galleryPersonFeatureDataReceiveQueue = deque(maxlen=256)

        # Query基础数据
        self.queryPersonIds = []
        self.queryFeatureArray = None
        self.queryImageArrayList = []
        self.queryPersonTagsList = []
        self.queryColorArray = None

        self.attributeLabeler = attributeLabeler

        # 其他多线程配置
        self.trackMessagesQueueName = f"TrackMessages"
        # tracking data send threading
        self.redisConnTrack = redis.StrictRedis(host=REDIS_TRACK_CONN_HOST, port=REDIS_TRACK_CONN_PORT)
        self.trackingDataPackageSendGlobalQueue = deque()
        self.trackingDataPackageSendThread = Thread(target=self.sendCameraTrackingData)
        self.trackingDataPackageSendThread.start()

        self.galleryPersonFeatureDataReceiveThread = Thread(target=self.galleryPersonFeatureDataReceiveEnsemble)
        self.galleryPersonFeatureDataReceiveThread.start()

    def sendCameraTrackingData(self):
        while True:
            currentLength = len(self.trackingDataPackageSendGlobalQueue)
            currentDataList = []
            for posIndex in range(currentLength):
                popData = self.trackingDataPackageSendGlobalQueue.popleft() # timestamp 在第一个字段
                timeStamp = popData[0]
                deadTracks = popData[1]
                cameraId = popData[3]
                resolution = popData[4]

                people = [{"id": track.id, "pos": track.lastLocation} for track in popData[2]]
                toSend = {"cameraId": cameraId,
                          "timeStamp": timeStamp,
                          "dead": list(deadTracks),
                          "people": people,
                          "resolution": resolution}
                currentDataList.append(toSend)

            if len(currentDataList) > 0:
                self.redisConnTrack.rpush(self.trackMessagesQueueName, json.dumps(currentDataList))
            time.sleep(0.01)

    def galleryPersonFeatureDataReceiveEnsemble(self):
        while True:
            if len(self.galleryPersonFeatureDataReceiveQueue) < self.galleryPersonFeatureDataReceiveQueue.maxlen:
                for _galleryFeatureResultQueueName in self.galleryFeatureResultQueueNames:
                    galleryRawData = self.redisConn.lpop(_galleryFeatureResultQueueName)
                    if galleryRawData is None:
                        time.sleep(0.01)
                        continue

                    dataPackage = json.loads(galleryRawData)

                    # decompress featureBytes and ImageBytes
                    for k in dataPackage["featureFields"]:
                        # k 必含 Bytes 子字符串
                        dataPackage[k.replace("Bytes", "Array")] = [(compressedFeatureTuple2Array(*item) if item is not None else None) for item in dataPackage[k]]

                    for k in dataPackage["imageFields"]:
                        frameImageList = []
                        for personImageList in dataPackage[k]:
                            currentImageList = []
                            for personImage in personImageList:
                                currentImageList.append(imageBytes2ImageArray(personImage) if personImage is not None else None)
                            frameImageList.append(currentImageList)
                        dataPackage[k.replace("Bytes", "Array")] = frameImageList

                    self.galleryPersonFeatureDataReceiveQueue.append(dataPackage)
                    # print(f"added :{len(self.galleryPersonFeatureDataReceiveQueue)}")
            else:
                time.sleep(0.01)

    def fetchQueryPersonFeatureData(self):
        queryRawData = self.redisConn.lpop(self.queryFeatureResultQueueName)
        #  first process query data
        if queryRawData is not None:
            queryParsedData = json.loads(queryRawData)
            # queryImageArray 是LIST OF IMAGE ARRAY
            queryFeatureBytes, queryArrayMax, queryArrayMin, queryImageBytesList = queryParsedData["personFeatureBytes"], queryParsedData["arrayMax"], queryParsedData["arrayMin"], queryParsedData["queryImageBytesList"]
            if len(queryFeatureBytes) > 0 and len(queryImageBytesList) > 0:
                newFeatureArray = compressedFeatureTuple2Array(queryFeatureBytes, queryArrayMin, queryArrayMax)
                if self.colorFilterSwitch:
                    newQueryFeatureArray, newQueryColorArray = newFeatureArray[:, :512], newFeatureArray[:, 512:]
                else:
                    newQueryFeatureArray = newFeatureArray
                newQueryImageArrayList = [cv2.resize(imageBytes2ImageArray(imgBytes), (128, 256)) for imgBytes in
                                          queryImageBytesList]
                newQueryPersonIds = queryParsedData["queryPersonIds"]
                newQueryPersonTags = self.attributeLabeler.labelImageList(newQueryImageArrayList)
                dataSend2HistoryDataBase = queryRawData[:-1] + b', "tags": ' + bytes(json.dumps(newQueryPersonTags),
                                                                                     encoding="utf-8") + b'}'
                self.redisConn.rpush(self.queryFeatureAndTagsResultQueueName, dataSend2HistoryDataBase)
                self.queryPersonIds.extend(newQueryPersonIds)
                if self.queryFeatureArray is None:
                    self.queryFeatureArray: np.ndarray = newQueryFeatureArray
                    self.queryImageArrayList: list = newQueryImageArrayList
                    self.queryPersonTagsList = newQueryPersonTags
                    if self.colorFilterSwitch:
                        self.queryColorArray: np.ndarray = newQueryColorArray
                else:
                    self.queryFeatureArray = np.concatenate((self.queryFeatureArray, newQueryFeatureArray), axis=0)
                    self.queryImageArrayList.extend(newQueryImageArrayList)
                    self.queryPersonTagsList.extend(newQueryPersonTags)
                    if self.colorFilterSwitch:
                        self.queryColorArray = np.concatenate((self.queryColorArray, newQueryColorArray), axis=0)

    def doMyJob(self):
        if len(self.galleryPersonFeatureDataReceiveQueue) == 0:
            time.sleep(0.01)
            return

        dataPackage = self.galleryPersonFeatureDataReceiveQueue.popleft()

        toLabelTracks, seenCameraId = [], set()

        galleryFrameCamerasList = dataPackage["frameCamerasList"]
        for frameIndex, galleryFrameCamera in enumerate(galleryFrameCamerasList):
            # singleFrameDataPackage = {k[5:][:-4]: v[frameIndex] for k, v in dataPackage.items() if k.startswith("frame")}
            singleFrameDataPackage = {}
            for k, v in dataPackage.items():
                if k.startswith("frame"):
                    # print(k, len(v), len(galleryFrameCamerasList))
                    singleFrameDataPackage[k[5:][:-4]] = v[frameIndex]
            if galleryFrameCamera in self.cameraPool:
                cameraInstance: Camera = self.cameraPool[galleryFrameCamera]
            else:
                cameraInstance: Camera = Camera(galleryFrameCamera, self.reidTaskSwitch, self.trackTaskSwitch)
                self.cameraPool[galleryFrameCamera] = cameraInstance

            # 数据包中有重复的情况下，需要提前打一次label，即本数据包可能会被执行多次打label操作；反之，如果没有重复，则在最后打一次label
            if cameraInstance.id in seenCameraId:
                # 收集需要进行过滤/后处理的所有轨迹，再一次性丢进抽象出来的 后处理 过程中.
                self.attributeLabeler.labelTracks(toLabelTracks)
                toLabelTracks, seenCameraId = [], set()

            cameraInstance.doPackageJob(singleFrameDataPackage, self.queryFeatureArray, self.queryImageArrayList, self.queryPersonTagsList, self.queryPersonIds)

            # TODO: 使用属性标注神经网络对camera实例中的未被属性标注的tracks打tag
            cameraUnlabeledTracks = cameraInstance.unlabeledTracks
            if len(cameraUnlabeledTracks) > 0:
                toLabelTracks.extend(cameraUnlabeledTracks)
                seenCameraId.add(cameraInstance.id)

            # print(cameraInstance.cameraTracker.trackingDataPackageSendQueue)
            self.trackingDataPackageSendGlobalQueue.extend(cameraInstance.cameraTracker.trackingDataPackageSendQueue)
            cameraInstance.cameraTracker.trackingDataPackageSendQueue.clear()

        if len(toLabelTracks) > 0:
            self.attributeLabeler.labelTracks(toLabelTracks)


#  TODO: 将属性标注之类的行为抽象为 后处理 ，重构
class AttributeLabeler:

    def __init__(self, AttributeExtractorClass) -> None:
        super().__init__()
        self.attributeModel = AttributeExtractorClass()

    def labelTracks(self, toLabelTracks: list):
        trackLabels = self.attributeModel.extractAttributeData([track.lastImageArray for track in toLabelTracks])
        for track, trackLabel in zip(toLabelTracks, trackLabels):
            track.tag = trackLabel

    def labelImageList(self, imageList: list):
        imageLabels = self.attributeModel.extractAttributeData(imageList)
        return imageLabels


def ReIDFactory(registerDetectionProcessIds, timeSleep=0.001):
    print("reidFactory")
    registerAttributeExtractor = ATTRIBUTE_NAME_TO_CLASS["Dummy"]
    registerAttributeLabeler = AttributeLabeler(registerAttributeExtractor)
    registerQueryFeatureResultQueueName = f"Query_ReidFeature"
    registerQueryFeatureAndTagsResultQueueName = "QueryFeatureAndTags"
    registerProcessor = RealTimeProcessor(registerDetectionProcessIds, registerQueryFeatureResultQueueName, registerQueryFeatureAndTagsResultQueueName, registerAttributeLabeler)
    while True:
        registerProcessor.doMyJob()
        time.sleep(timeSleep)


if __name__ == "__main__":
    print("ReidLauncher")
    AttributeExtractor = ATTRIBUTE_NAME_TO_CLASS["Dummy"]
    attributeLabeler = AttributeLabeler(AttributeExtractor)
    # Register Cameras
    dataRootPath = 'Z:/数据/双流 ReID 测试用数据2/videos'
    # cameraIds = os.listdir(dataRootPath)
    cameraIds = list(str(item) for item in range(0, 60))
    # STAND ALONE 测试
    galleryFeatureResultQueueNumber = 2
    queryFeatureResultQueueName = f"Query_ReidFeature"
    queryFeatureAndTagsResultQueueName = "QueryFeatureAndTags"
    tmpProcessor = RealTimeProcessor(galleryFeatureResultQueueNumber, queryFeatureResultQueueName, queryFeatureAndTagsResultQueueName, attributeLabeler)
    while True:
        tmpProcessor.doMyJob()
        time.sleep(0.01)

    # cameraids = ["c" + str(i % 4) for i in range(4)]
    # galleryFeatureResultQueueName = f"Gallery_ReidFeature_{str.join('_', cameraids)}"
    # queryFeatureResultQueueName = f"QueryReIdFeatureQueue"
    # tmpProcessor = RealTimeProcessor(galleryFeatureResultQueueName, queryFeatureResultQueueName, cameraids)
    # tmpProcessor.doMyJob()
    # print("good job!")

