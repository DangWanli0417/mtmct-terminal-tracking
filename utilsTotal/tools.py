# -*- coding:utf-8 -*-
# @Author : Re
# @Time : 2020/11/12 13:34
import json


__all__ = ["json2Bytes", "json2BytesHighOrder"]


def json2Bytes(jsonString: str, bytesName, bytesData):
    clip = bytes(jsonString[1:], encoding="utf-8")
    toSendBytes = bytes(f'{{"{bytesName}":"', encoding="utf-8") + bytesData + b'",' + clip
    return toSendBytes


def json2BytesHighOrder(jsonString: str, bytesNameList, bytesDataList):
    clip = bytes(jsonString[1:], encoding="utf-8")
    leftConcatenateBytes = b'{'
    for bytesName, bytesData in zip(bytesNameList, bytesDataList):
        leftConcatenateBytes += bytes(f'"{bytesName}":', encoding="utf-8") + bytesData + b','
    toSendBytes = leftConcatenateBytes + clip
    return toSendBytes

