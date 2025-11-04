# -*- coding:utf-8 -*-
# @Author : Re
# @Time : 2020/11/16 10:20

import numpy as np
import redis
from collections import deque, namedtuple, defaultdict
import json
from utils.img_utils import *
from utils.reid_utils import ReIdMatcher
from typing import (Sequence, Tuple, List, Deque)
import time

USEFUL_ATTRIBUTES = ["LowerLength", "LowerType", "UpperColor", "LowerColor"]


class FeatureBlock:
    """
    # TODO: a description for this class
    """
    def __init__(self, featureArray, locationsList, timeStampsList, tagsList, cameraList, trackFeatureSize: int = 3, maxBlockSize=9999) -> None:
        super().__init__()
        print("A New Feature Block Generated.")
        self.trackFeatureSize = trackFeatureSize
        self.featureArray: np.ndarray = featureArray
        self.locationsList: list = locationsList
        self.timeStampsList: list = timeStampsList
        self.tagsList: List[dict, ...] = tagsList
        self.cameraList: list = cameraList
        self.maxBlockSize = maxBlockSize
        self.reidMatcher = ReIdMatcher()
        self.searchResults = None

    @property
    def blockSize(self):
        return self.featureArray.shape[0]

    @property
    def checkBlockSize(self):
        return self.blockSize < self.maxBlockSize

    @staticmethod
    def arrayIndex2TrackIndex(indexList):
        return [index // 3 for index in indexList]

    def update(self, featureArray, locationsList, timeStampsList, tagsList, cameraList):
        self.featureArray = np.concatenate((featureArray, self.featureArray), axis=0)
        self.locationsList.extend(locationsList)
        self.timeStampsList.extend(timeStampsList)
        self.tagsList.extend(tagsList)
        self.cameraList.extend(cameraList)

    def search(self, queryFeatArray, queryTags, queryPersonIds):
        queryIndex2DataBaseIndexList, _ = self.reidMatcher.matching(self.featureArray, queryFeatArray)
        searchResults = {}
        # tags checking
        for queryPersonIndex, queryMatchedDataBaseIndices in queryIndex2DataBaseIndexList.items():
            queryPersonTag = queryTags[queryPersonIndex]
            queryPersonId = queryPersonIds[queryPersonIndex]
            confirmed = defaultdict(list)
            trackIndexList = self.arrayIndex2TrackIndex(queryMatchedDataBaseIndices)
            matchedInfos = ((self.tagsList[index], self.locationsList[index], self.timeStampsList[index], self.cameraList[index]) for index in trackIndexList)
            for matchedTrackTag, matchedTrackLocation, matchedTrackTimeStamp, matchedTrackCamera in matchedInfos:
                tagChecks = [queryPersonTag[checkedAttribute] is None or matchedTrackTag[checkedAttribute] is None or (matchedTrackTag[checkedAttribute] == queryPersonTag[checkedAttribute]) for checkedAttribute in
                             USEFUL_ATTRIBUTES]
                if all(tagChecks):
                    confirmed[matchedTrackCamera].append((matchedTrackTimeStamp[0], matchedTrackTimeStamp[1], matchedTrackLocation[0], matchedTrackLocation[1]))

            searchResults[queryPersonId] = confirmed

        self.searchResults = searchResults
        return searchResults


class HistoryTrackDataBase:

    def __init__(self, blockSliceNumber=33, blockNumber=50, maxBlockSize=9999) -> None:
        super().__init__()
        if not maxBlockSize % blockSliceNumber == 0:
            raise ValueError("blockSliceNumber must be a divisor of maxBlockSize")
        self.redisConn = redis.StrictRedis()
        self.redisQueueName = f"HistoryTrackDataBase"
        self.queryFeatureQueueName = f"QueryFeatureAndTags"
        self.querySearchResultQueueName = "HistoryQuerySearchResult"

        # self.redisConn.delete(self.querySearchResultQueueName)

        self.featureBlockQueue: Deque[FeatureBlock, ...] = deque(maxlen=blockNumber)

        self.maxBlockSize = maxBlockSize
        self.cacheSize = self.maxBlockSize // blockSliceNumber

        self.rawDataCacheList = deque(maxlen=self.cacheSize)

    @staticmethod
    def decompressedBytes2Array(inputBytes, inputMax, inputMin):
        decompressedArray = compressedFeatureBytes2Array(inputBytes)
        decompressedArray = decompressedArray / 255.0 * (inputMax - inputMin) + inputMin
        return decompressedArray

    def receiveMsg(self, inputMsg):
        rawMsg = self.redisConn.lpop(inputMsg)
        return json.loads(rawMsg) if rawMsg is not None else rawMsg

    def updateFeatureArray(self, override=False):
        if len(self.rawDataCacheList) == 0:
            return
        processFlag = len(self.rawDataCacheList) == self.cacheSize or override
        if not processFlag:
            return
        trackFeatureArrayList = []
        trackLocationsList = []
        trackTimeStampList = []
        trackTagsList = []
        trackCameraIdList = []
        for receivedMsg in self.rawDataCacheList:
            # trackLocations和trackTimeStamp都只有起始和结束的位置/时间
            trackLocations, trackTimeStamp, trackFeatureBytes, trackTags, cameraId = receivedMsg["trackLocations"], receivedMsg["trackTimeStamp"], receivedMsg["personFeatureBytes"], receivedMsg["tags"], receivedMsg["cameraId"]
            # trackPersonFeatureArray.shape为(3, 512)
            trackPersonFeatureArray = compressedFeatureBytes2Array(trackFeatureBytes)
            for seq, item in zip([trackFeatureArrayList, trackLocationsList, trackTimeStampList, trackTagsList, trackCameraIdList], [trackPersonFeatureArray, trackLocations, trackTimeStamp, trackTags, cameraId]):
                seq.append(item)

        trackFeatureArray = np.concatenate(trackFeatureArrayList, axis=0)

        # lastFeatureBlock: FeatureBlock = self.featureBlockQueue[-1]
        lastFeatureBlockSize = self.featureBlockQueue[-1].blockSize if len(self.featureBlockQueue) > 0 else self.maxBlockSize
        if lastFeatureBlockSize + trackFeatureArray.shape[0] <= self.maxBlockSize:
            print("Last Feature Block Updated.")
            self.featureBlockQueue[-1].update(trackFeatureArray, trackLocationsList, trackTimeStampList, trackTagsList, trackCameraIdList)
        else:
            self.featureBlockQueue.append(FeatureBlock(trackFeatureArray, trackLocationsList, trackTimeStampList, trackTagsList, trackCameraIdList))

        self.rawDataCacheList.clear()

    def processTrackMsg(self):
        receivedMsg = self.receiveMsg(self.redisQueueName)
        if receivedMsg is None:
            return False
        # self.rawDataCacheList.append(receivedMsg)
        return True

    def processQueryMsg(self):
        queryRawData = self.receiveMsg(self.queryFeatureQueueName)
        if queryRawData is None:
            return None
        queryFeatureBytes, queryArrayMax, queryArrayMin, queryPersonIds, queryPersonTags = queryRawData["personFeatureBytes"], queryRawData["arrayMax"], queryRawData["arrayMin"], queryRawData["queryPersonIds"], queryRawData["tags"]
        queryFeatureArray = self.decompressedBytes2Array(queryFeatureBytes, queryArrayMax, queryArrayMin)
        return queryFeatureArray, queryPersonTags, queryPersonIds

    def search(self, queryFeatureArray, queryPersonTags, queryPersonIds):
        globalSearchResults = {queryPersonId: defaultdict(list) for queryPersonId in queryPersonIds}
        for featureBlock in self.featureBlockQueue:
            blockSearchResults = featureBlock.search(queryFeatureArray, queryPersonTags, queryPersonIds)
            for blockQueryId, blockQueryCameraResult in blockSearchResults.items():
                for cameraId, trackInfos in blockQueryCameraResult.items():
                    globalSearchResults[blockQueryId][cameraId].extend(trackInfos)

        return globalSearchResults

    def sendDatum(self, trackSearchResult):
        self.redisConn.rpush(self.querySearchResultQueueName, trackSearchResult)

    def doMyJob(self):
        self.processTrackMsg()
        time.sleep(0.01)
        # overrideFlag = False
        # queryMsgRawData: tuple = self.processQueryMsg()
        # if queryMsgRawData is not None:
        #     overrideFlag = True
        # self.updateFeatureArray(overrideFlag)
        # if queryMsgRawData is not None:
        #     queryFeatureArray, queryPersonTags, queryPersonIds = queryMsgRawData
        #     globalSearchResults = self.search(queryFeatureArray, queryPersonTags, queryPersonIds)
        #     for queryPersonId, cameraId2Results in globalSearchResults.items():
        #         if len(cameraId2Results) > 0:
        #             self.sendDatum(json.dumps({"queryPersonId": queryPersonId, "cameraId2Results": cameraId2Results}))


if __name__ == "__main__":
    print("HistoryTrackDataBase")
    historyTrackDataBase = HistoryTrackDataBase()
    while True:
        historyTrackDataBase.doMyJob()








