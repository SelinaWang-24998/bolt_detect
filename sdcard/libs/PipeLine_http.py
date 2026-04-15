import os
from media.sensor import *
from media.display import *
from media.media import *
from libs.Utils import ScopedTiming
import nncase_runtime as nn
import image
import time


class PipeLine:
    """
    项目专用 PipeLine：
    - CHN0: RGB565，专门给网页推流/可选显示
    - CHN2: RGB888_PLANAR，专门给 AI 推理
    """

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
            else:
                Display.init(Display.LT9611, osd_num=self.osd_layer_num, to_ide=to_ide)

            self.display_size = [Display.width(), Display.height()]
            self.display_inited = True
            return True
        except Exception as e:
            print("[PipeLine_http] 显示初始化失败，自动切换为无屏模式:", e)
            self.display_inited = False
            self.enable_display = False
            return False

    def create(self, sensor=None, hmirror=None, vflip=None, fps=60, to_ide=True):
        with ScopedTiming("init PipeLine_http", self.debug_mode > 0):
            nn.shrink_memory_pool()
            if self.display_mode == "nt35516":
                fps = 30

            brd = os.uname()[-1]
            if brd in ("k230d_canmv_bpi_zero", "k230_canmv_lckfb", "k230d_canmv_atk_dnk230d"):
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

            # CHN0 直接使用 RGB565，和 test_video 的稳定推流路径保持一致。
            self.sensor.set_framesize(
                width=self.display_size[0], height=self.display_size[1], chn=CAM_CHN_ID_0
            )
            self.sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_0)

            # CHN2 专门给 AI 输入。
            self.sensor.set_framesize(
                width=self.rgb888p_size[0], height=self.rgb888p_size[1], chn=CAM_CHN_ID_2
            )
            self.sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)

            self.osd_img = image.Image(
                self.display_size[0], self.display_size[1], image.ARGB8888
            )

            MediaManager.init()
            self.sensor.run()

            if self.display_inited:
                try:
                    sensor_bind_info = self.sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)
                    Display.bind_layer(**sensor_bind_info, layer=Display.LAYER_VIDEO1)
                except Exception as e:
                    print("[PipeLine_http] 绑定显示层失败:", e)

    def get_frame(self):
        with ScopedTiming("get ai frame", self.debug_mode > 0):
            self.cur_frame = self.sensor.snapshot(chn=CAM_CHN_ID_2)
            return self.cur_frame.to_numpy_ref()

    def get_stream_frame(self):
        with ScopedTiming("get stream frame", self.debug_mode > 0):
            return self.sensor.snapshot(chn=CAM_CHN_ID_0)

    def show_image(self):
        if not self.display_inited:
            return
        with ScopedTiming("show result", self.debug_mode > 0):
            Display.show_image(self.osd_img, 0, 0, Display.LAYER_OSD3)

    def get_display_size(self):
        return self.display_size

    def destroy(self):
        with ScopedTiming("deinit PipeLine_http", self.debug_mode > 0):
            os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
            self.sensor.stop()
            if self.display_inited:
                Display.deinit()
                time.sleep_ms(50)
            MediaManager.deinit()
