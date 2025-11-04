
import threading
import re
import ffmpeg
rtspVideoPath=[]
class Video(threading.Thread):

    def __init__(self, cameraId,videopath):
        threading.Thread.__init__(self)
        self.cameraId=cameraId
        self.server_url="rtsp://10.10.10.111/"+str(cameraId)
        self.videopath=videopath
        self.video_format="rtsp"

    def run(self):
        rtspVideoPath.append(self.server_url)
        process = (
            ffmpeg
                .input(self.videopath)
                .output(
                self.server_url,
                rtsp_transport="tcp",
                vcodec="h264_nvenc",
                acodec="acc",
                f=self.video_format)
                .global_args("-re")  # argument to act as a live stream
                .run()
        )


# videoPathList=["D://testVideo/camera1.mp4", "D://testVideo/camera2.mp4"]
# videoPathList=["F://airportDatas/sample1.mp4", "F://airportDatas/sample2.mp4"]
videoPathList=["F://airportDatas/156_76_new.mp4", "F://airportDatas/156_77_new.mp4"]
# videoPathList=["***","***".....]
def PushRtspVideo(videoPathList):
    for pos in range(len(videoPathList)):
        video = Video(pos,videoPathList[pos])
        video.start()
    return (rtspVideoPath)

print(PushRtspVideo(videoPathList))


