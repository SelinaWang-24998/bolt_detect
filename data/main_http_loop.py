"""
单循环版 HTTP + 四模型螺栓检测推流：
- 使用你的 A/B/C 四模型级联逻辑
- 保留当前 RT-Smart Web 图传与前端控制
- 自动兼容 /sdcard 和 /sdcard/2026.4 两种模型部署路径
"""

import gc
import utime as time

from libs.PipeLine import PipeLine
from bolt_detector import BoltDetector

from rtsmart_web_adapter import RTWebAdapter
from detection_manager import DetectionManager
from wifi_config import WIFI_PASSWORD, WIFI_SSID, connect_wifi


RGB888P_SIZE = [640, 480]
DISPLAY_MODE = "lcd"
CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.45
FRAME_PUSH_INTERVAL_MS = 100
DETECTION_SAVE_THRESHOLD = 0.3
MAX_CONSECUTIVE_FAILURES = 5
RECORD_CHECK_INTERVAL_MS = 1000


def save_detection_records(results, detection_manager, save_img, threshold):
    if save_img is None or results is None:
        return

    try:
        for result in results:
            confidence = float(result.get("confidence", 0.0))
            if confidence < threshold:
                continue
            bbox = result.get("bbox", [0, 0, 0, 0])
            try:
                rec_id = detection_manager.add_detection(image=save_img, bbox=bbox, confidence=confidence)
                if rec_id:
                    print("[检测管理] 已保存记录 id={}, 置信度={:.2f}, bbox={}".format(rec_id, confidence, bbox))
            except Exception as e:
                print("[检测管理] 保存检测记录异常: {}".format(e))
    except Exception as e:
        print("[检测管理] 保存检测记录失败: {}".format(e))

def get_stream_image(pl, detection_enabled=False, overlay_osd=True):
    """
    优先从 CAM_CHN_ID_0 抓取用于 MJPEG 推流的图像。

    说明：
    - `pl.cur_frame` 来自 PipeLine 的 CHN2/RGB888P 推理通道，个别固件下直接用于
      Web 推流会出现整体发黑或只有极弱亮度变化的问题。
    - 参考项目的稳定做法是推流时单独从显示/视频通道再抓一张图，再按需叠加 OSD。
    """
    try:
        img = getattr(pl, "cur_frame", None)
#        from media.sensor import CAM_CHN_ID_0, CAM_CHN_ID_2

#        img = None

#        try:

#            img = pl.sensor.snapshot(chn=CAM_CHN_ID_0)
#        except Exception:
#            img = None

#        if img is None:
#            img = getattr(pl, "cur_frame", None)

#        if img is None:
#            try:
#                img = pl.sensor.snapshot(chn=CAM_CHN_ID_2)
#            except Exception:
#                img = None

        if img is None:
            return None

        if hasattr(img, "to_rgb888"):
            try:
                converted = img.to_rgb888()
                if converted is not None:
                    img = converted
            except Exception:
                pass

        if overlay_osd and detection_enabled and getattr(pl, "osd_img", None) is not None:
            try:
                img.draw_image(pl.osd_img, 0, 0, alpha=256)
            except Exception:
                pass
        return img
    except Exception as e:
        print("[HTTP] 获取 stream 图像失败: {}".format(e))
        return None

def get_stream_image1(pl, detection_enabled=False, overlay_osd=True):
    """
    获取用于 HTTP 推流的图像。
    优先使用 CHN2（RGB888P），避免 CHN0(YUV420) 直接压缩导致的偏色/花屏。
    """
    try:
        from media.sensor import CAM_CHN_ID_0, CAM_CHN_ID_2

        # CHN2 在 PipeLine 中配置为 RGB_888_PLANAR，更适合 JPEG 推流
        try:
            img = pl.sensor.snapshot(chn=CAM_CHN_ID_2)
        except Exception:
            img = pl.sensor.snapshot(chn=CAM_CHN_ID_0)

        # 显式转换为 RGB888，规避底层像素格式差异导致的颜色异常
        if hasattr(img, "to_rgb888"):
            try:
                converted = img.to_rgb888()
                if converted is not None:
                    img = converted
            except Exception:
                pass

        if overlay_osd and detection_enabled and getattr(pl, "osd_img", None) is not None:
            try:
                img.draw_image(pl.osd_img, 0, 0, alpha=256)
            except Exception:
                pass
        return img
    except Exception as e:
        print("[HTTP] 获取 stream 图像失败: {}".format(e))
        return None


def init_web_adapter(quality=50, debug_verbose=False):
    try:
        return RTWebAdapter(
            quality=quality,
            control_poll_interval_ms=5000,
            use_http_api_for_control=False,
            min_push_interval_ms=100,
            debug_verbose=debug_verbose,
        )
    except TypeError:
        print("[HTTP] ⚠️ 使用兼容模式初始化RTWebAdapter（旧版本不支持新参数）")
        web = RTWebAdapter(quality=quality, debug_verbose=debug_verbose)
        if hasattr(web, "set_control_poll_interval"):
            web.set_control_poll_interval(5000)
        if hasattr(web, "set_min_push_interval"):
            web.set_min_push_interval(100)
        return web

def main():
    print("=" * 60)
    print("HTTP + 四模型螺栓检测单线程推流")
    print("=" * 60)

    wifi_ok, ip_address = connect_wifi(WIFI_SSID, WIFI_PASSWORD)
    if not wifi_ok:
        print("[Wi-Fi] 连接失败，退出")
        return

    import rtsmart_web

    rtsmart_web.start_server()
    print("[HTTP] 服务器已启动")

    web = init_web_adapter(quality=60, debug_verbose=True)

    detection_manager = DetectionManager(save_dir="/data/detections", max_records=100)
    detection_manager.set_web_adapter(web)
    print("[检测管理] 检测记录管理器已初始化")

    pl = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode=DISPLAY_MODE)
    pl.create()
    display_size = pl.get_display_size()

    detector = BoltDetector(
        rgb888p_size=RGB888P_SIZE,
        display_size=display_size,
        conf_thresh=CONF_THRESHOLD,
        nms_thresh=NMS_THRESHOLD,
    )

    detection_enabled = False
    stream_enabled = True
    total_frames = 0
    total_detections = 0
    last_stats_ts = time.ticks_ms()
    last_frames = 0
    last_record_check_ts = time.ticks_ms()
    consecutive_failures = 0

    web.update_runtime(stream_enabled, detection_enabled, detector.det_conf_thresh)
    print("[提示] 浏览器访问 http://{}:8080/".format(ip_address))

    try:
        while True:
            try:
                frame = pl.get_frame()
                consecutive_failures = 0
                total_frames += 1
            except RuntimeError as e:
                consecutive_failures += 1
                if consecutive_failures <= 3:
                    print("[PipeLine] ⚠️ get_frame()失败 (连续{}次): {}".format(consecutive_failures, e))
                elif consecutive_failures == MAX_CONSECUTIVE_FAILURES:
                    print("[PipeLine] ❌ get_frame()连续失败{}次，可能传感器异常".format(MAX_CONSECUTIVE_FAILURES))
                time.sleep_ms(50)
                continue
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures <= 3:
                    print("[PipeLine] ⚠️ get_frame()异常: {}".format(e))
                time.sleep_ms(50)
                continue

            results = []
            if detection_enabled:
                results = detector.run(frame)
                detector.draw_result(results, pl.osd_img)
                total_detections += len(results)

                try:
                    from media.sensor import CAM_CHN_ID_0

                    save_img = pl.sensor.snapshot(chn=CAM_CHN_ID_0)
                    save_detection_records(results, detection_manager, save_img, DETECTION_SAVE_THRESHOLD)
                except Exception as e:
                    if total_detections <= 3:
                        print("[检测管理] 获取snapshot失败: {}".format(e))
            else:
                pl.osd_img.clear()

            pl.show_image()

            if stream_enabled:
                try:
                    stream_img = get_stream_image(pl, detection_enabled=detection_enabled, overlay_osd=True)
                    if stream_img is not None:
                        web.update_frame(stream_img)
                        if total_frames <= 3:
                            try:
                                print("[HTTP] ✅ 使用 helper 获取并推流, 尺寸: {}x{}".format(stream_img.width(), stream_img.height()))
                            except Exception:
                                print("[HTTP] ✅ 使用 helper 获取并推流")
                    elif total_frames % 30 == 0:
                        print("[HTTP] ⚠️ 无法获取有效的image.Image对象用于推流")
                except Exception as err:
                    if total_frames % 30 == 0:
                        print("[HTTP] 推帧失败：{}".format(err))
            now = time.ticks_ms()
            if time.ticks_diff(now, last_stats_ts) >= FRAME_PUSH_INTERVAL_MS:
                elapsed = max(time.ticks_diff(now, last_stats_ts) / 1000, 0.001)
                fps = (total_frames - last_frames) / elapsed
                web.update_stats_remote(total_frames, total_detections, fps)
                last_stats_ts = now
                last_frames = total_frames

            ctrl = web.pull_control()
            if ctrl:
                desired_stream = bool(ctrl.get("camera_desired"))
                desired_det = bool(ctrl.get("detection_desired"))
                desired_conf = ctrl.get("confidence_desired", detector.det_conf_thresh)

                if desired_stream != stream_enabled:
                    stream_enabled = desired_stream
                    print("[HTTP] 摄像头流状态 -> {}".format("开启" if stream_enabled else "暂停"))

                if desired_det != detection_enabled:
                    detection_enabled = desired_det
                    print("[HTTP] 检测状态 -> {}".format("开启" if detection_enabled else "关闭"))

                if isinstance(desired_conf, (int, float)) and 0.01 <= desired_conf <= 0.99:
                    detector.set_conf_threshold(desired_conf)

                web.update_runtime(stream_enabled, detection_enabled, detector.det_conf_thresh)

            if time.ticks_diff(now, last_record_check_ts) >= RECORD_CHECK_INTERVAL_MS:
                last_record_check_ts = now

            gc.collect()

    except KeyboardInterrupt:
        print("\n[系统] 捕获 Ctrl+C，正在退出...")
    finally:
        try:
            detector.deinit()
        except Exception:
            pass
        try:
            pl.destroy()
        except Exception:
            pass

        web.update_runtime(stream_enabled, detection_enabled, detector.det_conf_thresh)
        print("[系统] 已清理资源，程序结束")


if __name__ == "__main__":
    main()
