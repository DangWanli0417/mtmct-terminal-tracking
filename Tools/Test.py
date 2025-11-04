# -*- coding: utf-8 -*-
# @Author  : gl
# @Software: gczx
# @Contact : genglong@caacsri.com
# @Time    : 2021/8/5 16:53
import ffmpeg

process = (
            ffmpeg
                .input("C:/gczx/ffmpeg-20200315-c467328-win64-static/test/%d.jpg")
                .output(
                "C:/gczx/ffmpeg-20200315-c467328-win64-static/kk.mp4",
                r="1")
                                 .run()
        )