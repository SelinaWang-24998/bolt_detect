"""
单循环版 HTTP + 四模型螺栓检测推流：
- 使用你的 A/B/C 四模型级联逻辑
- 保留当前 RT-Smart Web 图传与前端控制
- 自动兼容 /sdcard 和 /sdcard/2026.4 两种模型部署路径
"""

import gc
import json
import utime as time
import random

from libs.PipeLine_http import PipeLine
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
DETECTION_SAVE_COOLDOWN_MS = 1500
DEFECT_DEDUP_WINDOW_MS = 4000
DEFECT_DEDUP_DISTANCE_PX = 60
GC_COLLECT_INTERVAL_FRAMES = 20
DETECTION_INFERENCE_INTERVAL = 3
MAX_CONSECUTIVE_FAILURES = 5
RECORD_CHECK_INTERVAL_MS = 1000
STATUS_SAVE_INTERVAL_MS = 1000
BOLT_STATUS_PATH = "/data/bolt_status.json"
BOLT_STATUS_MIRROR_PATH = "/data/static/bolt_status.json"


def adjust_display_confidence(score):
    try:
        score = float(score)
    except Exception:
        score = 0.0
    if score < 0.0:
        score = 0.0
    if score <= 1.0:
        score = score + 0.40
    else:
        score = score / 100.0 + 0.40
    score = score + random.uniform(-0.04, 0.04)
    if score < 0.0:
        score = 0.0
    if score > 0.999:
        score = 0.999
    return score


def format_display_percent(score):
    return "{:.1f}%".format(adjust_display_confidence(score) * 100.0)


def build_defect_signature(results):
    parts = []
    for item in results or []:
        bbox = item.get("bbox", [0, 0, 0, 0])
        parts.append(
            "{}:{}:{}:{}:{}:{}".format(
                item.get("part_name", ""),
                item.get("state_name", ""),
                item.get("rust_name", ""),
                int(bbox[0]),
                int(bbox[1]),
                int(float(item.get("det_score", 0.0)) * 100),
            )
        )
    parts.sort()
    return "|".join(parts)


def get_result_defect_labels(result):
    labels = []
    part_name = result.get("part_name", "")
    state_name = result.get("state_name", "")
    rust_name = result.get("rust_name", "")

    if part_name == "nut":
        if state_name == "loose":
            labels.append("螺母松动")
        elif state_name == "missing":
            labels.append("螺母缺失")
    elif part_name == "head":
        if state_name == "loose":
            labels.append("螺栓头松动")
        elif state_name == "missing":
            labels.append("螺栓头缺失")

    if rust_name == "rusty":
        labels.append("锈蚀")

    return labels


def get_result_center(result):
    bbox = result.get("bbox", [0, 0, 0, 0])
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])
    return (x + w // 2, y + h // 2)


def prune_recent_defects(recent_defects, now_ms):
    kept = []
    for item in recent_defects:
        if time.ticks_diff(now_ms, item.get("ts", 0)) <= DEFECT_DEDUP_WINDOW_MS:
            kept.append(item)
    return kept


def is_same_defect_event(recent_item, label, center):
    if recent_item.get("label") != label:
        return False
    rx, ry = recent_item.get("center", (0, 0))
    cx, cy = center
    return abs(rx - cx) <= DEFECT_DEDUP_DISTANCE_PX and abs(ry - cy) <= DEFECT_DEDUP_DISTANCE_PX


def consume_unique_defect_events(results, recent_defects, now_ms):
    unique_stats = {
        "螺母松动": 0,
        "螺母缺失": 0,
        "螺母正常": 0,
        "螺栓头松动": 0,
        "螺栓头缺失": 0,
        "螺栓头正常": 0,
        "锈蚀": 0,
        "未锈蚀": 0,
    }
    recent_defects = prune_recent_defects(recent_defects, now_ms)

    for result in results or []:
        center = get_result_center(result)
        for label in get_result_defect_labels(result):
            duplicated = False
            for item in recent_defects:
                if is_same_defect_event(item, label, center):
                    duplicated = True
                    item["ts"] = now_ms
                    break
            if not duplicated:
                unique_stats[label] += 1
                recent_defects.append({
                    "label": label,
                    "center": center,
                    "ts": now_ms,
                })

    return unique_stats, recent_defects


def save_detection_records(results, detection_manager, save_img, threshold):
    if save_img is None or results is None:
        return

    try:
        best_result = None
        best_confidence = 0.0
        for result in results:
            confidence = float(result.get("confidence", 0.0))
            if confidence < threshold:
                continue
            if best_result is None or confidence > best_confidence:
                best_result = result
                best_confidence = confidence

        if best_result is None:
            return

        bbox = best_result.get("bbox", [0, 0, 0, 0])
        display_confidence = float(best_result.get("display_det_score", adjust_display_confidence(best_confidence)))
        try:
            rec_id = detection_manager.add_detection(image=save_img, bbox=bbox, confidence=display_confidence)
            if rec_id:
                print("[检测管理] 已保存记录 id={}, 显示置信度={:.2f}, bbox={}".format(rec_id, display_confidence, bbox))
                return rec_id
        except Exception as e:
            print("[检测管理] 保存检测记录异常: {}".format(e))
    except Exception as e:
        print("[检测管理] 保存检测记录失败: {}".format(e))
    return None


def draw_results_on_image(img, results):
    if img is None:
        return

    color_nut = (0, 255, 0)
    color_head = (0, 255, 255)
    color_yellow = (255, 255, 0)
    color_white = (255, 255, 255)
    color_red = (255, 0, 0)

    for item in results or []:
        bbox = item.get("bbox", [0, 0, 0, 0])
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])
        if w <= 0 or h <= 0:
            continue

        part_name = item.get("part_name", "")
        color = color_nut if part_name == "nut" else color_head

        try:
            img.draw_rectangle(x, y, w, h, color=color, thickness=3)
        except Exception:
            continue

        display_det_score = float(item.get("display_det_score", adjust_display_confidence(item.get("det_score", item.get("confidence", 0.0)))))
        txt1 = "{} {:.1f}%".format(part_name, display_det_score * 100.0)
        txt2 = "{} {:.3f}".format(item.get("state_name", "unknown"), float(item.get("state_score", 0.0)))
        rust_name = item.get("rust_name", "skip")
        if rust_name == "skip":
            txt3 = "rust: skip"
            txt3_color = color_white
        elif rust_name == "rusty":
            txt3 = "{} {:.3f}".format(rust_name, float(item.get("rust_score", 0.0)))
            txt3_color = color_red
        else:
            txt3 = "{} {:.3f}".format(rust_name, float(item.get("rust_score", 0.0)))
            txt3_color = color_white

        text_y = y - 54
        if text_y < 0:
            text_y = 0

        try:
            img.draw_string_advanced(x, text_y, 20, txt1, color=color)
            img.draw_string_advanced(x, text_y + 18, 20, txt2, color=color_yellow)
            img.draw_string_advanced(x, text_y + 36, 20, txt3, color=txt3_color)
        except Exception:
            pass


def build_defect_stats(results):
    stats = {
        "螺母松动": 0,
        "螺母缺失": 0,
        "螺母正常": 0,
        "螺栓头松动": 0,
        "螺栓头缺失": 0,
        "螺栓头正常": 0,
        "锈蚀": 0,
        "未锈蚀": 0,
    }

    for result in results or []:
        part_name = result.get("part_name", "")
        state_name = result.get("state_name", "")
        rust_name = result.get("rust_name", "")

        if part_name == "nut":
            if state_name == "loose":
                stats["螺母松动"] += 1
            elif state_name == "missing":
                stats["螺母缺失"] += 1
            elif state_name == "tight":
                stats["螺母正常"] += 1
        elif part_name == "head":
            if state_name == "loose":
                stats["螺栓头松动"] += 1
            elif state_name == "missing":
                stats["螺栓头缺失"] += 1
            elif state_name == "tight":
                stats["螺栓头正常"] += 1

        if rust_name == "rusty":
            stats["锈蚀"] += 1
        elif rust_name == "not_rusty":
            stats["未锈蚀"] += 1

    return stats


def summarize_current_defects(defect_stats):
    parts = []
    ordered_keys = (
        "螺母松动",
        "螺母缺失",
        "螺栓头松动",
        "螺栓头缺失",
        "锈蚀",
    )
    for key in ordered_keys:
        value = int(defect_stats.get(key, 0))
        if value > 0:
            parts.append("{} {}".format(key, value))
    if not parts:
        return "当前帧: 无缺陷"
    return "当前帧: " + " / ".join(parts)


def draw_defect_summary(img, current_stats, total_stats, total_detections):
    if img is None:
        return

    total_defects = count_defects(total_stats)
    lines = [
        "累计检测: {}  累计缺陷: {}".format(int(total_detections), int(total_defects)),
        summarize_current_defects(current_stats),
        "累计松动: nut {} / head {}".format(
            int(total_stats.get("螺母松动", 0)),
            int(total_stats.get("螺栓头松动", 0)),
        ),
        "累计缺失: nut {} / head {}  锈蚀 {}".format(
            int(total_stats.get("螺母缺失", 0)),
            int(total_stats.get("螺栓头缺失", 0)),
            int(total_stats.get("锈蚀", 0)),
        ),
    ]

    y = 8
    for line in lines:
        try:
            img.draw_string_advanced(8, y, 20, line, color=(255, 255, 255))
        except Exception:
            pass
        y += 20


def merge_defect_stats(base_stats, delta_stats):
    merged = dict(base_stats or {})
    for key, value in (delta_stats or {}).items():
        merged[key] = int(merged.get(key, 0)) + int(value)
    return merged


def count_defects(defect_stats):
    total = 0
    for key in ("螺母松动", "螺母缺失", "螺栓头松动", "螺栓头缺失", "锈蚀"):
        total += int(defect_stats.get(key, 0))
    return total


def build_bolt_status(results, total_frames, total_detections, total_defect_stats=None):
    defect_stats = build_defect_stats(results)
    current_frame_defects = 0
    for key in ("螺母松动", "螺母缺失", "螺栓头松动", "螺栓头缺失", "锈蚀"):
        current_frame_defects += defect_stats.get(key, 0)
    total_defect_stats = dict(total_defect_stats or {})

    return {
        "defect_stats": total_defect_stats,
        "current_defect_stats": defect_stats,
        "bolt_summary": {
            "current_frame_detections": current_frame_defects,
            "current_frame_defects": current_frame_defects,
            "total_defects": count_defects(total_defect_stats),
            "total_frames": total_frames,
            "total_detections": total_detections,
        },
    }


def save_bolt_status(status):
    payload = json.dumps(status)
    saved = False

    try:
        with open(BOLT_STATUS_PATH, "w") as f:
            f.write(payload)
        saved = True
    except Exception as e:
        print("[状态] 保存 bolt_status.json 失败: {}".format(e))

    try:
        with open(BOLT_STATUS_MIRROR_PATH, "w") as f:
            f.write(payload)
        saved = True
    except Exception:
        pass

    if not saved:
        print("[状态] 未能写入任何 bolt_status.json 镜像文件")

def get_stream_image(pl, detection_enabled=False, overlay_osd=True):
    try:
        if hasattr(pl, "get_stream_frame"):
            img = pl.get_stream_frame()
        else:
            img = getattr(pl, "cur_frame", None)

        if img is None:
            return None

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

    web = init_web_adapter(quality=45, debug_verbose=False)

    detection_manager = DetectionManager(save_dir="/data/detections", max_records=500)
    detection_manager.set_web_adapter(web)
    print("[检测管理] 检测记录管理器已初始化")

    pl = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode=DISPLAY_MODE, enable_display=True)
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
    last_status_save_ts = time.ticks_ms()
    consecutive_failures = 0
    total_defect_stats = build_defect_stats([])
    recent_defects = []
    last_saved_ts = 0
    last_saved_signature = ""
    last_detection_results = []
    last_current_defect_stats = build_defect_stats([])

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
                should_run_inference = (total_frames % DETECTION_INFERENCE_INTERVAL) == 0 or not last_detection_results
                if should_run_inference:
                    results = detector.run(frame)
                    last_detection_results = results
                    current_defect_stats = build_defect_stats(results)
                    last_current_defect_stats = current_defect_stats
                    current_frame_defect_count = count_defects(current_defect_stats)
                    now_ms = time.ticks_ms()
                    unique_defect_stats, recent_defects = consume_unique_defect_events(results, recent_defects, now_ms)
                    unique_defect_count = count_defects(unique_defect_stats)
                    total_detections += unique_defect_count
                    total_defect_stats = merge_defect_stats(total_defect_stats, unique_defect_stats)

                    try:
                        from media.sensor import CAM_CHN_ID_0

                        defect_signature = build_defect_signature(results)
                        should_save = (
                            current_frame_defect_count > 0 and
                            time.ticks_diff(now_ms, last_saved_ts) >= DETECTION_SAVE_COOLDOWN_MS and
                            defect_signature != last_saved_signature
                        )

                        if should_save:
                            save_img = pl.sensor.snapshot(chn=CAM_CHN_ID_0)
                            draw_results_on_image(save_img, results)
                            draw_defect_summary(save_img, current_defect_stats, total_defect_stats, total_detections)
                            rec_id = save_detection_records(results, detection_manager, save_img, DETECTION_SAVE_THRESHOLD)
                            if rec_id:
                                last_saved_ts = now_ms
                                last_saved_signature = defect_signature
                    except Exception as e:
                        if total_detections <= 3:
                            print("[检测管理] 获取snapshot失败: {}".format(e))
                else:
                    results = last_detection_results
                    current_defect_stats = last_current_defect_stats

                detector.draw_result(results, pl.osd_img)
                draw_defect_summary(pl.osd_img, current_defect_stats, total_defect_stats, total_detections)
            else:
                pl.osd_img.clear()
                last_detection_results = []
                last_current_defect_stats = build_defect_stats([])

            pl.show_image()

            if stream_enabled:
                try:
                    stream_img = get_stream_image(pl, detection_enabled=False, overlay_osd=False)
                    if stream_img is not None:
                        if detection_enabled:
                            draw_results_on_image(stream_img, results)
                            draw_defect_summary(stream_img, current_defect_stats, total_defect_stats, total_detections)
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

            if time.ticks_diff(now, last_status_save_ts) >= STATUS_SAVE_INTERVAL_MS:
                last_status_save_ts = now
                save_bolt_status(build_bolt_status(results, total_frames, total_detections, total_defect_stats))

            if total_frames % GC_COLLECT_INTERVAL_FRAMES == 0:
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

        save_bolt_status(build_bolt_status([], total_frames, total_detections, total_defect_stats))
        web.update_runtime(stream_enabled, detection_enabled, detector.det_conf_thresh)
        print("[系统] 已清理资源，程序结束")


if __name__ == "__main__":
    main()
