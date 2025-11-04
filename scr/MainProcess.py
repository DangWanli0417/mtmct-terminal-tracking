# -*- coding: utf-8 -*-
# @Time: 2021/3/15 14:44
# @Author: libenchao
# @Contact: libenchao@caacsri.com
# @FileName: MainProcess.py

from multiprocessing import Process
import json
from DetectLauncher import DetectionFactory
from PersonFeatureExtractorLauncher import PersonFeatureExtractorFactory
from ReidLauncher import ReIDFactory
import multiprocessing
import encodings.idna
import os, sys
from collections import deque, defaultdict
import time
import redis
# TODO: 实现后处理函数
from Tools.PeoplePassDataProcess import PassRoi


def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj

import torch.jit

torch.jit.script_method = script_method
torch.jit.script = script


def launchAllProcesses(dataPackage):
    cameraSettings: list = dataPackage["cameraSettings"]
    cameraId2Address = {cameraSetting["cameraId"]: cameraSetting["address"] for cameraSetting in cameraSettings}
    cameraIds = list(cameraId2Address.keys())
    detectionProcessNumber = dataPackage["detectionProcessNumber"]
    featureExtractorProcessNumber = dataPackage["featureExtractorProcessNumber"]
    reidProcessNumber = dataPackage["reidProcessNumber"]
    cudaNumber = dataPackage["cudaNumber"]
    detectionBatchSize = dataPackage["detectionBatchSize"]
    featureExtractorBatchSize = dataPackage["featureExtractorBatchSize"]
    detectionPinnedDevice = dataPackage["detectionPinnedDevice"]
    featureExtractorPinnedDevice = dataPackage["featureExtractorPinnedDevice"]

    # 错误处理
    if len(cameraIds) < detectionProcessNumber:
        raise ValueError("processNumber must be bigger than length of cameraIds!")
    if cudaNumber < 1:
        raise ValueError("cudaNumber must be a positive number!")
    if cudaNumber > detectionProcessNumber:
        raise ValueError("processNumber must be bigger than cudaNumber!")
    if detectionProcessNumber // featureExtractorProcessNumber < 1 and detectionProcessNumber // reidProcessNumber < 1:
        raise ValueError("detectionProcessNumber must be multiple of featureExtractorProcessNumber and reidProcessNumber!")

    # 进程管理池
    allProcessInstances = []

    # 将摄像头分配到目标检测进程，即对摄像头分组，与目标检测分组一致
    processPool = deque([processId for processId in range(detectionProcessNumber)])
    detectionProcess2CameraIds = defaultdict(list)
    for cameraId in cameraIds:
        if len(processPool) == 0:
            processPool = deque([processId for processId in range(detectionProcessNumber)])
        toUseProcessId = processPool.popleft()
        detectionProcess2CameraIds[toUseProcessId].append(cameraId)

    # 分配cuda、启动目标检测
    cudaPool = deque([f"cuda:{cudaId}" for cudaId in range(cudaNumber)]) if not detectionPinnedDevice else deque([detectionPinnedDevice])
    for processId in range(detectionProcessNumber):
        if len(cudaPool) == 0:
            cudaPool = deque([f"cuda:{cudaId}" for cudaId in range(cudaNumber)])
        toUseCudaDevice = cudaPool.popleft()
        toUseCameraIds = detectionProcess2CameraIds[processId]
        toUseCameraId2Address = {toUseCameraId: cameraId2Address[toUseCameraId] for toUseCameraId in toUseCameraIds}
        detectionProcessInstance = Process(target=DetectionFactory, args=(processId, toUseCameraId2Address, toUseCudaDevice, 16, detectionBatchSize, 0.01))
        detectionProcessInstance.start()
        allProcessInstances.append(detectionProcessInstance)
        print(f"Main Process {processId}, detection pid is {detectionProcessInstance.pid}")

    # 分配cuda和摄像头组，启动特征提取器
    cudaPool = deque([f"cuda:{cudaId}" for cudaId in range(cudaNumber)]) if not featureExtractorPinnedDevice else deque([featureExtractorPinnedDevice])
    detectionProcessPool = deque([processId for processId in range(detectionProcessNumber)])
    for processId in range(featureExtractorProcessNumber):
        stepMultiple = int(detectionProcessNumber / featureExtractorProcessNumber)
        detectionProcessIds = []
        tmpCameraIds = []
        for step in range(stepMultiple):
            detectionProcessId = detectionProcessPool.popleft()
            detectionProcessIds.append(detectionProcessId)
            tmpCameraIds.extend(detectionProcess2CameraIds[detectionProcessId])

        if len(cudaPool) == 0:
            cudaPool = deque([f"cuda:{cudaId}" for cudaId in range(cudaNumber)]) if not featureExtractorPinnedDevice else deque([featureExtractorPinnedDevice])
        toUseCudaDevice = cudaPool.popleft()

        personFeatureProcessInstance = Process(target=PersonFeatureExtractorFactory,
                                               args=(detectionProcessIds, tmpCameraIds, toUseCudaDevice,
                                                     featureExtractorBatchSize, 0.01))
        personFeatureProcessInstance.start()
        allProcessInstances.append(personFeatureProcessInstance)
        print(f"Main Process {processId}, personFeature pid is {personFeatureProcessInstance.pid}")

    # 摄像头组，启动ReID器
    detectionProcessPool = deque([processId for processId in range(detectionProcessNumber)])
    for processId in range(reidProcessNumber):
        stepMultiple = int(detectionProcessNumber / reidProcessNumber)
        detectionProcessIds = []
        for step in range(stepMultiple):
            detectionProcessIds.append(detectionProcessPool.popleft())
        reIDProcessInstance = Process(target=ReIDFactory, args=(detectionProcessIds, 0.01))
        reIDProcessInstance.start()
        allProcessInstances.append(reIDProcessInstance)
        print(f"Main Process {processId}, reid pid is {reIDProcessInstance.pid}")

    # 启动后处理程序
    postProcessInstance = Process(target=PassRoi, args=(cameraSettings , ))
    postProcessInstance.start()
    allProcessInstances.append(postProcessInstance)
    print(f"Post Process , Post pid is {postProcessInstance.pid}")
    # 返回所有多进程实例
    return allProcessInstances


def killAllProcess(processList):
    for process in processList:
        process.terminate()
    print("Killed all process")
    return []


if __name__ == "__main__":
    multiprocessing.freeze_support()
    with open("./configs/DefaultConfigs.json", 'r') as f:
        defaultConfigs = json.load(f)
    mainConfigsRedisHost, mainConfigsRedisPort, mainConfigsRedisName = defaultConfigs["MainConfigsRedisHost"], defaultConfigs["MainConfigsRedisPort"], defaultConfigs["MainConfigsRedisName"]
    redisConn = redis.StrictRedis(host=mainConfigsRedisHost, port=mainConfigsRedisPort)
    runningProcessList = []
    while True:
        if redisConn.llen(mainConfigsRedisName) > 0:
            if len(runningProcessList) > 0:
                runningProcessList = killAllProcess(runningProcessList)
            originalData = redisConn.lpop(mainConfigsRedisName)#
            dataPackage = json.loads(originalData)
            runningProcessList = launchAllProcesses(dataPackage)


        time.sleep(10)
