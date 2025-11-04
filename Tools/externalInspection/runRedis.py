'''
Author       : kev1n
Date         : 2021-07-05 11:29:14
LastEditTime : 2021-07-05 15:56:51
a2V2MW4uemhlbmdAb3V0bG9vay5jb20=: 
'''
import redis
import json

with open("./configs/DefaultConfigs.json", 'r') as f:
    defaultConfigs = json.load(f)

mainConfigsRedisHost, mainConfigsRedisPort = defaultConfigs["MainConfigsRedisHost"], defaultConfigs["MainConfigsRedisPort"]
redisServer = redis.StrictRedis(host=mainConfigsRedisHost, port=mainConfigsRedisPort)

Frames = json.loads(redisServer.lpop('Frames_TCDM_2'))
for Frame in Frames:
    print(Frame)
