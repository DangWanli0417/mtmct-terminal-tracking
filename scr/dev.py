# -*- coding: utf-8 -*-

import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2
import os
from ReidNetwork.resnet import resnet50, resnet50_attri
from ReidNetwork.osnet_ain import osnet_ain_x1_0
import torch
import cv2
from torch.backends import cudnn
from utils.backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video, STANDARD_COLORS, standard_to_bgr, \
    get_index_label, plot_one_box
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import math
import matplotlib.pyplot as plt
#import seaborn
import os
from collections import defaultdict, deque
from collections import Counter
from utils.reid_utils import resize_and_pad
from utils.img_utils import lowlight_process
from scipy.stats import spearmanr

# TODO: use PIL read image test the procedure...
from torchvision.transforms import Resize, Normalize, ToTensor
from torchvision import transforms




def getQueryArrayAndStdAndMeanAndBias(inputQueryArray):
    _std = np.std(inputQueryArray, axis=1,  keepdims=True)
    _mean = np.mean(inputQueryArray, axis=1, keepdims=True)
    _bias = inputQueryArray - _mean
    return inputQueryArray, _std, _mean, _bias


def getGalleryArrayAndStdAndMeanAndBias(inputGalleryArray):
    _std = np.expand_dims(np.std(inputGalleryArray, axis=1), axis=0)
    _mean = np.mean(inputGalleryArray, axis=1, keepdims=True)
    _bias = inputGalleryArray - _mean
    return inputGalleryArray, _std, _mean, _bias


def getPearsonArrayBeta(inputQueryStd, inputQueryBias, inputGalleryStd, inputGalleryBias):
    _cov = np.dot(inputQueryBias, inputGalleryBias.transpose())
    _stdArray = np.dot(inputQueryStd, inputGalleryStd)
    _corrArray: np.ndarray = _cov / _stdArray / 512
    return _corrArray


def getBetaMatrixMethod(inputQueryMatrix: np.ndarray, inputGalleryMatrix: np.ndarray):
    if inputQueryMatrix.ndim == 2:
        inputQueryMatrix = np.expand_dims(inputQueryMatrix, axis=1)
    if inputGalleryMatrix.ndim == 2:
        inputGalleryMatrix = np.expand_dims(inputGalleryMatrix, axis=0)
    _maxMatrix, _minMatrix = np.maximum(inputQueryMatrix, inputGalleryMatrix), np.minimum(inputQueryMatrix, inputGalleryMatrix)
    _localSimilarityMatrix = _minMatrix / _maxMatrix
    np.nan_to_num(_localSimilarityMatrix, copy=False, nan=1)
    _localSimilarityMatrix = np.mean(_localSimilarityMatrix, axis=2)
    return _localSimilarityMatrix


def getBeatMatrix(inputQueryArray, inputGalleryArray):
    stacked = np.empty((2, 512), dtype=np.float32)
    similarityMatrix = np.empty((inputQueryArray.shape[0], inputGalleryArray.shape[0]), dtype=np.float32)
    for queryIndex, queryVector in enumerate(inputQueryArray):
        for galleryIndex, galleryVector in enumerate(inputGalleryArray):
            similarityMatrix[queryIndex, galleryIndex] = getBetaSimilarity(queryVector, galleryVector)
    return similarityMatrix


def getBetaSimilarity(inputQuery, inputGallery):
    stack = np.stack([inputQuery, inputGallery])
    maxVector, minVector = np.max(stack, axis=0), np.min(stack, axis=0)
    similarityVector: np.ndarray = 1.0 - (maxVector - minVector) / (maxVector + 0.0000001)
    #np.fill
    similarityMean = np.mean(similarityVector)

    # _cov = np.dot(inputQueryBias, inputGalleryBias.transpose())
    # _stdArray = np.dot(inputQueryStd, inputGalleryStd)
    # _corrArray: np.ndarray = _cov / _stdArray / 512
    return similarityMean


def pre_processing(img, flipFlag=0):
    """
    :param img: cv2 image, np.ndarray
    :return:
    """
    # by opencv method
    # img = img[:,:,::-1]
    if flipFlag:
        img = cv2.flip(img, 1)
    #img_re = cv2.resize(img, (128, 256), interpolation=cv2.INTER_CUBIC)
    img_re = resize_and_pad(img)
    # cv2.imwrite('./enhance_before.jpg', img_re)
    # img_en = lowlight_process(img_re)
    # cv2.imwrite('./enhance_after.jpg', img_en)
    img_re = cv2.cvtColor(img_re, cv2.COLOR_BGR2RGB)
    img_re = img_re / 255.0
    img_norm = (img_re - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    return np.transpose(img_norm, (2, 0, 1))


def pre_processing_untran(img, flipFlag=0):
        """
        :param img: cv2 image, np.ndarray
        :return:
        """
        # by opencv method
        # img = img[:,:,::-1]
        if flipFlag:
            img = cv2.flip(img, 1)
        # img_re = cv2.resize(img, (128, 256), interpolation=cv2.INTER_CUBIC)
        img_re = resize_and_pad(img)
        # img_re = cv2.cvtColor(img_re, cv2.COLOR_BGR2RGB)
        # img_re = img_re / 255.0
        # img_norm = (img_re - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        return img_re


    # # by pil method
    # trans = transforms.Compose([
    #     Resize((256, 128)),
    #     ToTensor(),
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
    # )
    # return trans(img)


def display(preds, imgs, imgName):
    pedestrianImgs, pedestrianLocations = [], []
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            if preds[i]['class_ids'][j] == 0:
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                pedestrianLocations.append((x1, y1, x2, y2))
                obj = obj_list[preds[i]['class_ids'][j]]
                pedestrianImgs.append(imgs[i][int(y1):int(y2), int(x1):int(x2), :])
                score = float(preds[i]['scores'][j])
                plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                             color=color_list[get_index_label(obj, obj_list)])

        # cv2.imwrite(f'res/{imgName}.jpg', imgs[i])

    return pedestrianImgs, pedestrianLocations, imgs[0]


class FeatureExtractor:
    def __init__(self) -> None:
        super().__init__()
        self.model = osnet_ain_x1_0()
        self.batchSize = 32

    def extractBatchFeature(self, imageBatch: list, flipFlag) -> list:
        """
        :param imageBatch: List of images(np.ndarray)
        :return: list of features(np.ndarray 1-d)
        """
        image_shape = imageBatch[0].shape
        # image_shape = imageBatch[0].size
        batch_size = len(imageBatch)
        input_batch = np.zeros((batch_size, 3, 256, 128), dtype=np.float32)
        for idx, img in enumerate(imageBatch):
            img = pre_processing(img, flipFlag)
            input_batch[idx, :] = img

        input_batch = torch.from_numpy(input_batch)
        input_batch = input_batch.cuda(CUDA_NUMBER)
        self.model.eval()
        self.model.cuda(CUDA_NUMBER)
        with torch.no_grad():
            batchFeat_gpu = self.model(input_batch)
            batchFeat_cpu = batchFeat_gpu.cpu().numpy()
            batchFeat = []
            for i in range(batch_size):
                batchFeat.append(batchFeat_cpu[i, :])

            return batchFeat

    def extractListFeature(self, imageList, flipFlag=0):
        if len(imageList) == 0:
            raise ValueError("The length of imageList is 0.")

        image_num = len(imageList)
        imageFeat = []
        inference_times = image_num // self.batchSize
        extra_images = imageList[inference_times * self.batchSize:]
        for i in range(1, inference_times + 1):
            start = (i - 1) * self.batchSize
            end = i * self.batchSize
            cur_imageBatch = imageList[start:end]
            batchFeat = self.extractBatchFeature(cur_imageBatch, flipFlag)
            imageFeat.extend(batchFeat)

        if extra_images:
            extra_batchFeat = self.extractBatchFeature(extra_images, flipFlag)
            imageFeat.extend(extra_batchFeat)

        return imageFeat

    @staticmethod
    def preprocessImages(imageList):
        batch_size = len(imageList)
        input_batch = np.zeros((batch_size, 256, 128, 3), dtype=np.float32)
        for idx, _img in enumerate(imageList):
            _img = pre_processing_untran(_img)
            input_batch[idx, :] = _img
        return input_batch

def query_src_pre(query_dir):
    res1D = []
    person_list = os.listdir(query_dir)
    filename2PersonId = {}
    for p in person_list:
        person_files_list = os.listdir(os.path.join(query_dir, p))
        person_imgs_list = [query_dir + '/' + p + '/' + tmpFile for tmpFile in person_files_list if (tmpFile.endswith('.jpg') or tmpFile.endswith('.png'))]
        for f in person_imgs_list:
            filename2PersonId[f] = p

        #res2D.append(person_imgs_list)

        res1D.extend(person_imgs_list)
    return res1D, filename2PersonId


class ReIdTracker:

    def __init__(self, initialId, initialTimeStamp, initialFeature, initialLocation) -> None:
        super().__init__()
        self.id = initialId
        self.lastLocation = initialLocation
        self.lastCenter = (int((self.lastLocation[0] + self.lastLocation[2]) / 2), int((self.lastLocation[1] + self.lastLocation[3]) / 2))
        self.lastTimestamp = 0
        self.lastFeature = initialFeature
        self.appearTime = 0
        self.idCounter = {}
        self.scoreMap = {}
        self.matchedTime = 0
        self.disappearTime = 0
        self.betaSimilarity = 0.
        self.lastPearson = 0.

    def __str__(self):
        return str(self.id)

    def update(self, newLocation, newFeature, timeStampSecond):
        lastWidth = self.lastLocation[2] - self.lastLocation[0]
        newCenter = (int((newLocation[0] + newLocation[2]) / 2), int((newLocation[1] + newLocation[3]) / 2))
        displacement = math.sqrt(math.pow(self.lastCenter[0] - newCenter[0], 2) + math.pow(self.lastCenter[1] - newCenter[1], 2))
        normalizedDisplacement = displacement / lastWidth / (timeStampSecond - self.lastTimestamp)
        if normalizedDisplacement < 2:
            self.lastLocation, self.lastCenter, self.lastFeature = newLocation, newCenter, newFeature
            self.appearTime += 1
            self.disappearTime = 0
            self.lastTimestamp = timeStampSecond
        else:
            self.disappearTime += 1
        # positionOffset = (1 - 1 / math.exp(normalizedDisplacement - 2)) if normalizedDisplacement > 2 else 0
        # refinedFeatureSimilarity = featureSimilarity - positionOffset
        # if refinedFeatureSimilarity > 0.65:
        #     self.lastLocation, self.lastCenter = newLocation, newCenter

    def addMatchedId(self, matchedIds, matchedSimi):
        for enumId, matchedId in enumerate(matchedIds):
            _personId = pictureId2personId[matchedId]
            _personId = matchedId
            self.idCounter[_personId] = self.idCounter.get(_personId, 0) + 1
            self.scoreMap[_personId] = round((self.scoreMap.get(_personId, 0) * (self.idCounter[_personId] - 1) + matchedSimi[enumId]) / self.idCounter[_personId], 4)
        self.matchedTime += 1
        # self.idCounter.update(matchedIds)

    @property
    def confirmedId(self):
        if self.matchedTime <= 0: # REID matched times, 表示与query里面的person匹配到的次数，当前帧与query匹配到了多个人时候，也只算做匹配到了一次。
            return None
        return self.scoreMap
        # if self.matchedTime <= 0:
        #     return None
        # maxScoreKey = max(self.scoreMap.items(), key=lambda item: item[1])
        # # print(maxScoreKey)
        # return maxScoreKey
        # mostCommon = self.idCounter.most_common()
        # if len(mostCommon) == 0:
        #     return None
        # ratio = self.idCounter[mostCommon[0][1]] / self.matchedTime
        # if ratio >= 0.5:
        #     return mostCommon[0][1]
        # else:
        #     return None


if __name__ == '__main__':
    # Video Path
    CUDA_NUMBER = 0
    #video_src = "E:/WorkSpace/数据/双流机场 ReID 测试用数据/录像/时间段1 CA4501 F 摄像头 1-2-3/摄像头2-172.22.153.4 - L2-EF-候机厅（4803）-2020-10-19_13：52-16：27.mp4"
    src_list = ["L2-E-E指廊通道(1081)-2020-09-04_10h00min00s000ms_1.asf",
                "DR.172.22.170.18 - L2-E-144休息区(5337)-2020-09-04_10h00min00s000ms.mp4",
                "DR.172.22.158.13 - L2-E-152djk全景 (4922)-2020-09-04_10h00min00s000ms.mp4",
                "414-DE-L2-DE连廊头等舱对面-F03F003-0701-3(1652)-2020-09-04_09h59min14s453ms.mp4",
                "BC.L2-DE-出港厅(5694)-2020-09-04_10h00min00s000ms.mp4",
                "172.22.188.17 - L2-DE-出港通道(5715)-2020-09-04_10h00min00s000ms.asf",
                "BC.L2-D-D指廊根部(740)-2020-09-04_10h00min00s000ms.mp4",
                "BC.172.22.189.16 - L2-D-129djk全景(5727)-2020-09-04_10h00min00s000ms.mp4",
                "DR.172.22.182.7 - L2-D-137djk全景(5435)-2020-09-04_10h00min00s000ms.mp4"]
    save_list = ["1081", "5337","4922","1652", "5694", "5715", "740", "5727", "5435"]
    # query_dir = "./query/CA4501"
    #rootPath = "E:/WorkSpace/"
    rootPath = "Z:/"
    for i in range(1, 2):
        # if i < 2:
        #     continue
        #video_src = rootPath + "数据/双流机场 ReID 测试用数据/录像/时间段1 CA4501 F 摄像头 1-2-3/" + src_list[i]
        video_src = rootPath + "数据/双流视频/20200904/" + src_list[i]
        save_path = rootPath + "Code/videoResult-girlAndGuy/" + save_list[i]
    # video_src = "Z:/数据/双流机场 ReID 测试用数据/录像/时间段1 CA4501 F 摄像头 1-2-3/摄像头2-172.22.153.4 - L2-EF-候机厅（4803）-2020-10-19_13：52-16：27.mp4"
    #TODO: add query_dir
        query_dir = "./query/girlAndGuy"
        # save_path = "Z:/Code/videoResult/test-1029-2"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        trackIdDeque = deque(range(0, 400), maxlen=400)
        resultPath = './res'
        resultFile = './' + video_src.split('/')[-1] + '.txt'
        toDrawBetaSimi, toDrawPearson = [], []

        # Basic Settings
        compound_coef = 3
        force_input_size = None  # set None to use default size
        threshold = 0.6
        iou_threshold = 0.4
        use_cuda = True
        use_float16 = False
        cudnn.fastest = True
        cudnn.benchmark = True
        obj_list = ['person']

        reid_threshold = 0.45

        # tf bilinear interpolation is different from any other's, just make do
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

        # load model
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        color_list = standard_to_bgr(STANDARD_COLORS)
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
        model.load_state_dict(torch.load('efficientdet-d3_0_84000.pth'))
        model.requires_grad_(False)
        model.eval()

        # Video capture
        cap = cv2.VideoCapture(video_src)

        frameId = 0

        if use_cuda:
            model = model.cuda(CUDA_NUMBER)
        if use_float16:
            model = model.half()

        featureExtractor = FeatureExtractor()

        # read query image
        queryImagesList = [] # cv2 type
        # _queryImagesList = []  # cv2 type
        queryImagesPath, queryImages2PersonDict = query_src_pre(query_dir)
        for img in queryImagesPath:
            # TODO: 这里有个问题是，之前读入query为png格式，图像为四通道数据，但是深度学习reid网络没有报错，对最后的效果似乎有一些影响，
            # TODO：
            queryImgTmp = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
            # _queryImgTmp = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
            if queryImgTmp is None:
                raise ValueError
            queryImagesList.append(queryImgTmp)
            # _queryImagesList.append(_queryImgTmp)
        pictureId2personId = {pictureId: queryImages2PersonDict[filePath] for pictureId, filePath in enumerate(queryImagesPath)}
        # pictureId2fileName = {pictureId: filePath for pictureId, filePath in enumerate(queryImagesPath)}
        #queryHeight, queryWidth = queryImage.shape[0], queryImage.shape[1]
        # _queryFeature = np.stack(featureExtractor.extractListFeature(_queryImagesList))
        queryFeature = np.stack(featureExtractor.extractListFeature(queryImagesList))
        # featDiff = queryFeature - _queryFeature
        queryImagesList = featureExtractor.preprocessImages(queryImagesList)
        queryArray, queryStd, queryMean, queryBias = getQueryArrayAndStdAndMeanAndBias(queryFeature)
        resultDict = defaultdict(list)
        # lastGallery, lastTrackerMap = None, None
        lastTrackPool = set()

        while True:
            newTrackPool = set()
            ret, frame = cap.read() # frame 读入为rgb顺序
            if not ret:
                print(video_src, "video read error!!!")
                break

            frameId += 1
            # if frameId <= 4500:
            #     continue

            if frameId % 15 != 0:
                continue
            # frame preprocessing
            frameH, frameW = frame.shape[0], frame.shape[1]
            timePoint = int(frameId / 15)
            ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda(CUDA_NUMBER) for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

            # model predict

            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)

            # result
            out = invert_affine(framed_metas, out)
            _detectedPedestrianImages, _detectedPedestrianLocations, annotatedImages = display(out, ori_imgs, str(frameId))
            detectedPedestrianLocations, detectedPedestrianImages = [], []
            for detectedId, location in enumerate(_detectedPedestrianLocations):
                if (location[3] - location[1]) * (location[2] - location[0]) >= 6000:
                    detectedPedestrianLocations.append(location)
                    detectedPedestrianImages.append(_detectedPedestrianImages[detectedId])

            if len(detectedPedestrianLocations) == 0:
                # for save the picture without detections
                #cv2.imwrite(f'{save_path}/{timePoint}.jpg', annotatedImages)
                continue

            reidFeatures = np.stack(featureExtractor.extractListFeature(detectedPedestrianImages))
            # reidFeatures_fliped = np.stack(featureExtractor.extractListFeature(detectedPedestrianImages, 1))

            if len(lastTrackPool) == 0:
                for detectedPedestrianId, detectedPedestrianLocation in enumerate(detectedPedestrianLocations):
                    lastTrackPool.add(ReIdTracker(trackIdDeque.popleft(), timePoint, reidFeatures[detectedPedestrianId], detectedPedestrianLocation))

                # for save the picture without tracklet
                #cv2.imwrite(f'{save_path}/{timePoint}.jpg', annotatedImages)
                continue

            lastTrackArray, trackIds = [], []

            for track in lastTrackPool:
                lastTrackArray.append(track.lastFeature)
                trackIds.append(track)

            lastTrackArray = np.stack(lastTrackArray)
            trackArray, trackStd, trackMean, trackBias = getQueryArrayAndStdAndMeanAndBias(lastTrackArray)
            galleryArray, galleryStd, galleryMean, galleryBias = getGalleryArrayAndStdAndMeanAndBias(reidFeatures)
            # galleryArray_flip, galleryStd_flip, galleryMean_flip, galleryBias_flip = getGalleryArrayAndStdAndMeanAndBias(reidFeatures_fliped)
            pearsonMatrix = getPearsonArrayBeta(trackStd, trackBias, galleryStd, galleryBias) # 当前跟踪使用的相似度计算方式，pearson
            #myTrackBetaMatrix = getBetaMatrixMethod(trackArray, galleryArray) #将跟踪使用的相似度修改为自定义的计算方式


            reidPearsonMatrix = getPearsonArrayBeta(queryStd, queryBias, galleryStd, galleryBias)
            # reidBetaMatrix = getBeatMatrix(queryArray, galleryArray)
            reidBetaMatrix = getBetaMatrixMethod(queryArray, galleryArray)
            # reidBetaMatrix_fliped = getBeatMatrix(queryArray, galleryArray_flip)
            # reidBetaMatrix = np.min(np.stack((reidBetaMatrix, reidBetaMatrix_fliped)), axis=0)
            # print(np.max(reidBetaMatrix))
            # for galleryEnumId, galleryVector in enumerate(galleryArray):
            #     for trackEnumId, trackVector in enumerate(lastTrackArray):
            #         betaSimilarity = getBetaSimilarity(trackVector, galleryVector)
            #         if betaSimilarity > 0.65:

            # reid
            matchedPersonIndices, matchedCurrentIndices = np.where(reidBetaMatrix > reid_threshold)
            currentId2MatchedPersonId, currentId2MatchedScore = defaultdict(list), defaultdict(list)
            for numId, reidMatchedCurrentIndex in enumerate(matchedCurrentIndices):
                currentId2MatchedPersonId[reidMatchedCurrentIndex].append(matchedPersonIndices[numId])
                currentId2MatchedScore[reidMatchedCurrentIndex].append(reidBetaMatrix[matchedPersonIndices[numId], reidMatchedCurrentIndex])

            # tracking
            bestMatchedIndex = pearsonMatrix.argmax(axis=1)
            usedCurrentIds, unMatchedTracks = set(), []
            for lastTrackId, currentId in enumerate(bestMatchedIndex):
                pearSimilarity = pearsonMatrix[lastTrackId, currentId]
                track = trackIds[lastTrackId]
                if pearSimilarity >= 0.65 and currentId not in usedCurrentIds:
                    usedCurrentIds.add(currentId)
                    track.betaSimilarity = round(float(getBetaSimilarity(reidFeatures[currentId], track.lastFeature)), 4)
                    track.lastPearson = round(float(pearSimilarity), 4)
                    # print(track.betaSimilarity)
                    track.update(detectedPedestrianLocations[currentId],
                                 reidFeatures[currentId],
                                 timePoint)
                    # tag track with person id
                    if len(currentId2MatchedPersonId[currentId]) > 0:
                        track.addMatchedId(currentId2MatchedPersonId[currentId], currentId2MatchedScore[currentId])
                else:
                    track.disappearTime += 1
                    unMatchedTracks.append(track)

            unusedCurrentIds = set(range(0, len(detectedPedestrianLocations))).difference(usedCurrentIds)
            for unusedCurrentId in unusedCurrentIds:
                lastTrackPool.add(ReIdTracker(trackIdDeque.popleft(), timePoint, reidFeatures[unusedCurrentId],
                                              detectedPedestrianLocations[unusedCurrentId]))

            for unMatchedTrack in unMatchedTracks:
                if unMatchedTrack.disappearTime >= 3:
                    trackIdDeque.append(unMatchedTrack.id)
                    lastTrackPool.remove(unMatchedTrack)

            #################################################

            # draw result

            everReIdMatched = False
            everMatchedReIdPictureIds = set()
            for nowTrack in lastTrackPool:
                if nowTrack not in unMatchedTracks:
                    coord = nowTrack.lastLocation
                    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
                    confirmedInfo = nowTrack.confirmedId
                    if confirmedInfo is not None:
                        everReIdMatched = True
                        everMatchedReIdPictureIds.update(confirmedInfo.keys())
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
            for everMatchedReIdPictureId in everMatchedReIdPictureIds:
                if displayColumnNumber >= 8:
                    displayRowNumber += 1
                    displayColumnNumber = 0
                displayAreaHeightStart = (displayCharHeight + displayCharHeight) * displayRowNumber
                displayAreaWidthStart = displayPictureWidth * displayColumnNumber
                matchedReIdImage = queryImagesList[everMatchedReIdPictureId]
                annotatedImages[displayAreaHeightStart:displayAreaHeightStart + displayPictureHeight, displayAreaWidthStart:displayAreaWidthStart + displayPictureWidth, :] = matchedReIdImage
                cv2.putText(annotatedImages, f"{everMatchedReIdPictureId}",
                            (displayAreaWidthStart + 64, displayAreaHeightStart + displayPictureHeight), 0, 0.5,
                            [255, 255, 255],
                            thickness=1, lineType=cv2.FONT_HERSHEY_SIMPLEX)
                displayColumnNumber += 1
            # toDrawBetaSimi.extend(reidBetaMatrix.flatten().tolist())
            # toDrawPearson.extend(reidPearsonMatrix.flatten().tolist())
            # if int(frameId / 15) == 1500:
            #     seaborn.jointplot(x=toDrawBetaSimi, y=toDrawPearson, kind="hex", color="#4CB391")
            #     plt.show()

            # save image, only save those reid matched frames.
            if everReIdMatched:
                print(timePoint)
                pass
                cv2.imwrite(f'{save_path}/{timePoint}.jpg', annotatedImages)

