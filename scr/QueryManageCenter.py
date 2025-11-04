# -*- coding:utf-8 -*-
# @Author : Re
# @Time : 2020/12/7 12:13
from typing import Mapping, overload, Iterable, Tuple

import numpy as np
from collections import defaultdict
from redis import StrictRedis
import json


class QueryManager:

    def __init__(self):
        self.queryHistorySearchQueueName = f"HistoryQuerySearchResult"
        self.queryRealtimeSearchQueueName = f"RealtimeQueryMatchResult"
        self.querySendQueueName = f"Query_InputImageBytes"
        self.queryReceiveQueueName = f"Query_RawDataPackage"
        self.redisConn = StrictRedis()
        self.queryId2QueryInstance = {}

        self.checkFiledList = ["queryImageBytesList", "queryPersonIds"]

    def receiveQueryHistorySearchMsg(self):
        queryHistorySearchResult = self.redisConn.lpop(self.queryHistorySearchQueueName)
        if queryHistorySearchResult is None:
            return None
        queryHistorySearchResult = json.loads(queryHistorySearchResult)
        return queryHistorySearchResult

    def receiveQueryRealtimeReidMsg(self):
        queryRealtimeReidResult = self.redisConn.lpop(self.queryRealtimeSearchQueueName)
        if queryRealtimeReidResult is None:
            return None
        queryRealtimeReidResult = json.loads(queryRealtimeReidResult)
        return queryRealtimeReidResult

    def receiveQueryRawDataPackage(self):
        queryRawDataBytes = self.redisConn.lpop(self.queryReceiveQueueName)
        if queryRawDataBytes is None:
            return False
        parsedData = json.loads(queryRawDataBytes)
        if all([field in parsedData for field in self.checkFiledList]):
            return queryRawDataBytes

    def sendQueryImageData(self, queryRawDataBytes):
        self.redisConn.rpush(self.querySendQueueName, queryRawDataBytes)

    def updateQuery(self, queryId, querySearchResults):
        if queryId not in self.queryId2QueryInstance:
            self.queryId2QueryInstance[queryId] = Query(queryId)
        toUpdateQuery = self.queryId2QueryInstance[queryId]
        toUpdateQuery.update(querySearchResults)

    def doMyJob(self):
        queryRawDataBytes = self.receiveQueryRawDataPackage()
        if queryRawDataBytes:
            self.sendQueryImageData(queryRawDataBytes)
        queryHistorySearchResult = self.receiveQueryHistorySearchMsg()
        if queryHistorySearchResult is not None:
            self.updateQuery(queryHistorySearchResult["queryPersonId"], queryHistorySearchResult["cameraId2Results"])

        queryRealtimeSearchResult = self.receiveQueryRealtimeReidMsg()
        if queryRealtimeSearchResult is not None:
            queryPersonIds = queryRealtimeSearchResult["queryPersonIds"]
            for queryPersonId in queryPersonIds:
                self.updateQuery(queryPersonId, queryRealtimeSearchResult["cameraId2Results"])


class Query:

    def __init__(self, queryId) -> None:
        super().__init__()
        self.id = queryId
        self.cameraId2TrackInfos = defaultdict(list)

    def __str__(self) -> str:
        return self.id

    def update(self, querySearchResults: dict):
        for cameraId, trackInfos in querySearchResults.items():
            self.cameraId2TrackInfos[cameraId].extend(trackInfos)

    @property
    def data(self):
        return json.dumps(self.cameraId2TrackInfos)


if __name__ == "__main__":
    import time
    print("QueryManangeCenter")
    tempManager = QueryManager()
    while True:
        tempManager.doMyJob()
        time.sleep(0.01)






