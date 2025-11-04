# -*- coding: utf-8 -*-
# @Author  : gl
# @Software: gczx
# @Contact : genglong@caacsri.com
# @Time    : 2021/8/17 10:17
import time
from Tools.PeoplePassDataProcess.PointInRoi import isPointinPolygon


class CameraRule():

    def __init__(self, cameraRuleId, confInfo, mode) -> None:
        super().__init__()
        self.cameraRuleId = cameraRuleId
        self.ROI = confInfo['ROI']
        self.system = confInfo['system']
        self.ruleId = confInfo['ruleId']
        self.startTime =time.time()
        self.mode = mode
        self.w = 1280
        self.h = 720
        self.result = {"roi": None, "passIds": [], "passCount": 0, "dailyCount": 0, "count": 0}

    def initData(self, resolution):
        self.w = resolution[1]
        self.h = resolution[0]
        PushRoi = []
        CurrentRoi = self.ROI
        if len(CurrentRoi) <= 2:
            PushRoi = [[0, 0], [self.w, 0], [self.w, self.h], [0, self.h], [0, 0]]
        else:
            for i in CurrentRoi:
                points = i.split(",")
                x = int(float(points[0]) * self.w)
                y = int(float(points[1]) * self.h)
                PushRoi.append([x, y])

        self.result['roi'] = PushRoi


    def upDate(self, people, deadList):

        # 过线结果
        if self.mode == "passCount":
            for location in people:
                midPointPos = [(location['pos'][0] + location['pos'][2]) / 2.0, location['pos'][1] + location['pos'][3] / 2.0]
                if str(location['id']) not in self.result["passIds"]:
                    isInRoi = False
                    for i in range(4):
                        if isPointinPolygon(location['pos'][i], self.result["roi"]):
                            isInRoi = True
                            break
                    if isInRoi:
                        self.result["passIds"].append(str(location['id']))
                        self.result["passCount"] += 1
                        self.result["dailyCount"] += 1
                for deadId in deadList:
                    if str(deadId) in self.result["passIds"]:
                        self.result["passIds"].remove(str(deadId))

        # 行人计数结果/车辆计数
        if self.mode == "trafficCount" or self.mode == "peopleCount":
            for location in people:
                midPointPos = [(location['pos'][0] + location['pos'][2]) / 2.0,
                               location['pos'][1] + location['pos'][3] / 2.0]
                isInRoi = False
                for i in range(4):
                    if isPointinPolygon(location['pos'][i], self.result["roi"]):
                        isInRoi = True
                        break
                if isInRoi:
                    self.result["count"] += 1
