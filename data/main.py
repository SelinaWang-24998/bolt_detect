from libs.PipeLine import PipeLine
from libs.Utils import ScopedTiming
from bolt_detector import BoltDetector
import gc


RGB888P_SIZE = [640, 480]
DISPLAY_MODE = "lcd"
CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.45


def main():
    pl = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode=DISPLAY_MODE)
    pl.create()
    display_size = pl.get_display_size()

    detector = BoltDetector(
        rgb888p_size=RGB888P_SIZE,
        display_size=display_size,
        conf_thresh=CONF_THRESHOLD,
        nms_thresh=NMS_THRESHOLD,
    )

    try:
        while True:
            with ScopedTiming("total", 1):
                frame = pl.get_frame()
                results = detector.run(frame)
                detector.draw_result(results, pl.osd_img)
                pl.show_image()
                gc.collect()
    finally:
        detector.deinit()
        pl.destroy()


if __name__ == "__main__":
    main()
