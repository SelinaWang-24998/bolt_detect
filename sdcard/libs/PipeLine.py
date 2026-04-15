import os
import ujson
from media.sensor import *
from media.display import *
from media.media import *
from libs.Utils import ScopedTiming
import nncase_runtime as nn
import ulab.numpy as np
import image
import gc
import sys
import time


class PipeLine:
    def __init__(
        self,
        rgb888p_size=[224, 224],
        display_mode="hdmi",
        display_size=None,
        osd_layer_num=1,
        debug_mode=0,
        enable_display=None,
    ):
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        if display_size is None:
            self.display_size = None
        else:
            self.display_size = [display_size[0], display_size[1]]
        self.display_mode = display_mode
        self.sensor = None
        self.osd_img = None
        self.cur_frame = None
        self.debug_mode = debug_mode
        self.osd_layer_num = osd_layer_num
        self.display_inited = False

        # 当前板子图传优先，所以默认允许无屏运行。
        if enable_display is None:
            self.enable_display = False
        else:
            self.enable_display = bool(enable_display)

    def _safe_init_display(self, to_ide=True):
        if not self.enable_display:
            return False

        try:
            if self.display_mode == "hdmi":
                if self.display_size is None:
                    Display.init(Display.LT9611, osd_num=self.osd_layer_num, to_ide=to_ide)
                else:
                    Display.init(
                        Display.LT9611,
                        width=self.display_size[0],
                        height=self.display_size[1],
                        osd_num=self.osd_layer_num,
                        to_ide=to_ide,
                    )
            elif self.display_mode == "lcd":
                if self.display_size is None:
                    Display.init(Display.ST7701, osd_num=self.osd_layer_num, to_ide=to_ide)
                else:
                    Display.init(
                        Display.ST7701,
                        width=self.display_size[0],
                        height=self.display_size[1],
                        osd_num=self.osd_layer_num,
                        to_ide=to_ide,
                    )
            elif self.display_mode == "lt9611":
                if self.display_size is None:
                    Display.init(Display.LT9611, osd_num=self.osd_layer_num, to_ide=to_ide)
                else:
                    Display.init(
                        Display.LT9611,
                        width=self.display_size[0],
                        height=self.display_size[1],
                        osd_num=self.osd_layer_num,
                        to_ide=to_ide,
                    )
            elif self.display_mode == "st7701":
                if self.display_size is None:
                    Display.init(Display.ST7701, osd_num=self.osd_layer_num, to_ide=to_ide)
                else:
                    Display.init(
                        Display.ST7701,
                        width=self.display_size[0],
                        height=self.display_size[1],
                        osd_num=self.osd_layer_num,
                        to_ide=to_ide,
                    )
            elif self.display_mode == "hx8399":
                if self.display_size is None:
                    Display.init(Display.HX8399, osd_num=self.osd_layer_num, to_ide=to_ide)
                else:
                    Display.init(
                        Display.HX8399,
                        width=self.display_size[0],
                        height=self.display_size[1],
                        osd_num=self.osd_layer_num,
                        to_ide=to_ide,
                    )
            elif self.display_mode == "nt35516":
                if self.display_size is None:
                    Display.init(Display.NT35516, osd_num=self.osd_layer_num, to_ide=to_ide)
                else:
                    Display.init(
                        Display.NT35516,
                        width=self.display_size[0],
                        height=self.display_size[1],
                        osd_num=self.osd_layer_num,
                        to_ide=to_ide,
                    )
            else:
                Display.init(Display.LT9611, osd_num=self.osd_layer_num, to_ide=to_ide)

            self.display_size = [Display.width(), Display.height()]
            self.display_inited = True
            return True
        except Exception as e:
            print("[PipeLine] 显示初始化失败，自动切换为无屏模式:", e)
            self.display_inited = False
            self.enable_display = False
            return False

    def create(self, sensor=None, hmirror=None, vflip=None, fps=60, to_ide=True):
        with ScopedTiming("init PipeLine", self.debug_mode > 0):
            nn.shrink_memory_pool()
            if self.display_mode == "nt35516":
                fps = 30

            brd = os.uname()[-1]
            if brd == "k230d_canmv_bpi_zero":
                self.sensor = Sensor(fps=30) if sensor is None else sensor
            elif brd == "k230_canmv_lckfb":
                self.sensor = Sensor(fps=30) if sensor is None else sensor
            elif brd == "k230d_canmv_atk_dnk230d":
                self.sensor = Sensor(fps=30) if sensor is None else sensor
            else:
                self.sensor = Sensor(fps=fps) if sensor is None else sensor

            self.sensor.reset()
            if hmirror is not None and (hmirror is True or hmirror is False):
                self.sensor.set_hmirror(hmirror)
            if vflip is not None and (vflip is True or vflip is False):
                self.sensor.set_vflip(vflip)

            self._safe_init_display(to_ide=to_ide)

            if self.display_size is None:
                self.display_size = [self.rgb888p_size[0], self.rgb888p_size[1]]

            # 无屏模式下，不再要求通道0必须匹配显示面板。
            self.sensor.set_framesize(w=self.display_size[0], h=self.display_size[1])
            self.sensor.set_pixformat(PIXEL_FORMAT_YUV_SEMIPLANAR_420)

            self.sensor.set_framesize(
                w=self.rgb888p_size[0], h=self.rgb888p_size[1], chn=CAM_CHN_ID_2
            )
            self.sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)

            self.osd_img = image.Image(
                self.display_size[0], self.display_size[1], image.ARGB8888
            )

            if self.display_inited:
                sensor_bind_info = self.sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)
                Display.bind_layer(**sensor_bind_info, layer=Display.LAYER_VIDEO1)

            MediaManager.init()
            self.sensor.run()

    def get_frame(self):
        with ScopedTiming("get a frame", self.debug_mode > 0):
            self.cur_frame = self.sensor.snapshot(chn=CAM_CHN_ID_2)
            input_np = self.cur_frame.to_numpy_ref()
            return input_np

    def show_image(self):
        if not self.display_inited:
            return
        with ScopedTiming("show result", self.debug_mode > 0):
            Display.show_image(self.osd_img, 0, 0, Display.LAYER_OSD3)

    def get_display_size(self):
        return self.display_size

    def destroy(self):
        with ScopedTiming("deinit PipeLine", self.debug_mode > 0):
            os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
            self.sensor.stop()
            if self.display_inited:
                Display.deinit()
                time.sleep_ms(50)
            MediaManager.deinit()
