# -*- coding: utf-8 -*-
# @Author  : gl
# @Software: gczx
# @Contact : genglong@caacsri.com
# @Time    : 2021/8/13 10:30
import json
import redis
from Tools.PeoplePassDataProcess.CameraRule import CameraRule



with open("./configs/DefaultConfigs.json", 'r') as f:
    defaultConfigs = json.load(f)
detectTarget = defaultConfigs["Mode"]

countTask = defaultConfigs["CountTaskSwitch"]
trackTask = defaultConfigs["TrackTaskSwitch"]
mode = defaultConfigs["Mode"]

if countTask and not trackTask:
    if detectTarget == "Person":
        mode = "peopleCount"
    elif detectTarget == "Car":
        mode = "trafficCount"
    else:
        mode = None


if trackTask and not countTask:
    if detectTarget == "Person":
        mode = "passCount"
    elif detectTarget == "Car":
        mode = "passCount"
    else:
        mode = None

cameraIdRuleIdSeparator = defaultConfigs["CameraIdRuleIdSeparator"]
detectorResultRedisName = defaultConfigs["DetectorResultRedisName"]
mainConfigsRedisHost, mainConfigsRedisPort = defaultConfigs["MainConfigsRedisHost"], defaultConfigs[
    "MainConfigsRedisPort"]
resultConfigsRedisHost, resultConfigsRedisPort = defaultConfigs["ResultBackRedisHost"], defaultConfigs[
    "ResultBackRedisPort"]
redisServer = redis.StrictRedis(host=mainConfigsRedisHost, port=mainConfigsRedisPort)
resultRedisServer = redis.StrictRedis(host=resultConfigsRedisHost, port=resultConfigsRedisPort)


def afterProcess(CameraSettings: list):  # list of dict
    print("afterProcess is started")
    CameraDic={}
    cameraRule2Obj={}
    for cameraSet in CameraSettings:
        cameraRuleId = cameraSet['cameraId']
        if cameraRuleId not in cameraRule2Obj:
            cameraRule2Obj[cameraRuleId] = CameraRule(cameraRuleId, cameraSet['Post'][0], mode)

    while True:
        if redisServer.llen(detectorResultRedisName) > 0:
            trackPackages = json.loads(redisServer.lpop(detectorResultRedisName))
            # 过线结果
            if mode == "passCount":
                for trackPackage in trackPackages:
                    cameraId_ruleId = trackPackage["cameraId"]
                    if cameraId_ruleId in cameraRule2Obj:
                        cameraRule2Obj[cameraId_ruleId].initData(trackPackage['resolution'])
                        cameraRule2Obj[cameraId_ruleId].upDate(trackPackage['people'], trackPackage['dead'])
                        Pushdata = {"system": cameraRule2Obj[cameraId_ruleId].system, "ruleId": cameraId_ruleId, "algorithmType": mode,
                                    "count": cameraRule2Obj[cameraId_ruleId].result['passCount'], "timeStamp": trackPackage["timeStamp"],
                                    "todayAmount": cameraRule2Obj[cameraId_ruleId].result['dailyCount'],
                                    'startTime': cameraRule2Obj[cameraId_ruleId].startTime,
                                    'endTime': trackPackage["timeStamp"]}
                        # 结果输出到redis队列
                        resultRedisServer.rpush(defaultConfigs["ResultBackRedisName"], json.dumps(Pushdata))
                        cameraRule2Obj[cameraId_ruleId].result['passCount'] = 0

            # 行人计数结果/车辆计数
            if mode == "trafficCount" or mode == "peopleCount":
                cameraId_ruleIds=trackPackages["cameras"]
                for i in range(len(cameraId_ruleIds)):
                    cameraId_ruleId = cameraId_ruleIds[i]
                    frameTimeStamp = trackPackages["frameTimeStampList"][i]
                    frameResolution = trackPackage["frameResolutionList"][i]
                    batchLocation = trackPackage["batchLocationsList"][i]
                    if cameraId_ruleId in cameraRule2Obj:
                        cameraRule2Obj[cameraId_ruleId].initData(frameResolution)
                        cameraRule2Obj[cameraId_ruleId].upDate(batchLocation, None)
                        Pushdata = {"system": cameraRule2Obj[cameraId_ruleId].system, "ruleId": cameraId_ruleId,
                                    "algorithmType": mode,
                                    "count": cameraRule2Obj[cameraId_ruleId].result['count'],
                                    "timeStamp": trackPackage["timeStamp"]}
                        # 结果输出到redis队列
                        resultRedisServer.rpush(defaultConfigs["ResultBackRedisName"], json.dumps(Pushdata))
                        cameraRule2Obj[cameraId_ruleId].result['count'] = 0
