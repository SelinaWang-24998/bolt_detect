import gc
import os
import time
import nncase_runtime as nn
import ulab.numpy as np


A_MODEL_NAME = "2ndA_best.kmodel"
B_NUT_MODEL_NAME = "B_nut_best.kmodel"
B_HEAD_MODEL_NAME = "B_head_best.kmodel"
C_RUST_MODEL_NAME = "2ndC_rust_best.kmodel"

A_INPUT_W = 640
A_INPUT_H = 640
CLS_INPUT_W = 224
CLS_INPUT_H = 224

A_LABELS = ["nut", "head"]
B_LABELS = ["loose", "missing", "tight"]
C_LABELS = ["not_rusty", "rusty"]

DET_CONF_THRESH = 0.15
DET_NMS_THRESH = 0.45
MAX_BOXES_NUM = 3
CLS_EVERY_N_FRAMES = 5
GC_EVERY_N_FRAMES = 40
ONLY_RUST_FOR_TIGHT = True

COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (0, 255, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
A_COLORS = [COLOR_GREEN, COLOR_CYAN]


def _path_exists(path):
    try:
        os.stat(path)
        return True
    except OSError:
        return False


def _resolve_model_path(name):
    candidates = (
        "/sdcard/" + name,
        "/sdcard/2026.4/" + name,
        "/data/" + name,
    )
    for path in candidates:
        if _path_exists(path):
            return path
    raise OSError("Kmodel file not exist: " + name)


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def argmax_1d(arr):
    max_idx = 0
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_idx, max_val


def nms(boxes, scores, thresh):
    order = list(range(len(boxes)))
    order.sort(key=lambda i: scores[i], reverse=True)
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        new_order = []
        x1_i, y1_i, x2_i, y2_i = boxes[i]
        area_i = (x2_i - x1_i + 1) * (y2_i - y1_i + 1)

        for j in order[1:]:
            x1_j, y1_j, x2_j, y2_j = boxes[j]
            area_j = (x2_j - x1_j + 1) * (y2_j - y1_j + 1)

            xx1 = max(x1_i, x1_j)
            yy1 = max(y1_i, y1_j)
            xx2 = min(x2_i, x2_j)
            yy2 = min(y2_i, y2_j)

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            inter = w * h
            denom = area_i + area_j - inter
            iou = 0.0 if denom <= 0 else inter / denom

            if iou < thresh:
                new_order.append(j)

        order = new_order

    return keep


def display_score_value(score):
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
    if score > 0.999:
        score = 0.999
    return score


def display_score_text(score):
    return "{:.1f}%".format(display_score_value(score) * 100.0)


class BoltDetector:
    def __init__(self, rgb888p_size, display_size, conf_thresh=DET_CONF_THRESH, nms_thresh=DET_NMS_THRESH):
        self.ai_w = rgb888p_size[0]
        self.ai_h = rgb888p_size[1]
        self.disp_w = display_size[0]
        self.disp_h = display_size[1]
        self.det_conf_thresh = float(conf_thresh)
        self.det_nms_thresh = float(nms_thresh)

        self.frame_id = 0
        self.last_cls_results = []

        self.kpu_a = None
        self.kpu_b_nut = None
        self.kpu_b_head = None
        self.kpu_c_rust = None

        self.ai2d_det = None
        self.ai2d_det_builder = None
        self.det_u8_out_tensor = None

        self.roi_input_shape = [1, 3, self.ai_h, self.ai_w]
        self.roi_output_shape = [1, 3, CLS_INPUT_H, CLS_INPUT_W]
        self.roi_u8_tensor = None
        self.roi_builder_cache_key = None
        self.roi_builder = None

        self.a_model_path = _resolve_model_path(A_MODEL_NAME)
        self.b_nut_model_path = _resolve_model_path(B_NUT_MODEL_NAME)
        self.b_head_model_path = _resolve_model_path(B_HEAD_MODEL_NAME)
        self.c_rust_model_path = _resolve_model_path(C_RUST_MODEL_NAME)

        print("[BoltDetector] A模型:", self.a_model_path)
        print("[BoltDetector] B螺母模型:", self.b_nut_model_path)
        print("[BoltDetector] B螺栓头模型:", self.b_head_model_path)
        print("[BoltDetector] C锈蚀模型:", self.c_rust_model_path)

        self.load_models()
        self.build_preprocess()

    def set_conf_threshold(self, value):
        if value < 0.01:
            value = 0.01
        if value > 0.99:
            value = 0.99
        self.det_conf_thresh = float(value)

    def load_models(self):
        self.kpu_a = nn.kpu()
        self.kpu_b_nut = nn.kpu()
        self.kpu_b_head = nn.kpu()
        self.kpu_c_rust = nn.kpu()

        self.kpu_a.load_kmodel(self.a_model_path)
        self.kpu_b_nut.load_kmodel(self.b_nut_model_path)
        self.kpu_b_head.load_kmodel(self.b_head_model_path)
        self.kpu_c_rust.load_kmodel(self.c_rust_model_path)

    def build_preprocess(self):
        self.ai2d_det = nn.ai2d()
        self.ai2d_det.set_dtype(
            nn.ai2d_format.NCHW_FMT,
            nn.ai2d_format.NCHW_FMT,
            np.uint8,
            np.uint8,
        )
        self.ai2d_det.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d_det_builder = self.ai2d_det.build(
            [1, 3, self.ai_h, self.ai_w],
            [1, 3, A_INPUT_H, A_INPUT_W],
        )

        self.det_u8_out_tensor = nn.from_numpy(np.ones((1, 3, A_INPUT_H, A_INPUT_W), dtype=np.uint8))
        self.roi_u8_tensor = nn.from_numpy(np.ones((1, 3, CLS_INPUT_H, CLS_INPUT_W), dtype=np.uint8))

    def get_roi_builder(self, x1, y1, crop_w, crop_h):
        key = (x1, y1, crop_w, crop_h)
        if self.roi_builder is not None and self.roi_builder_cache_key == key:
            return self.roi_builder

        ai2d_roi = nn.ai2d()
        ai2d_roi.set_dtype(
            nn.ai2d_format.NCHW_FMT,
            nn.ai2d_format.NCHW_FMT,
            np.uint8,
            np.uint8,
        )
        ai2d_roi.set_crop_param(True, x1, y1, crop_w, crop_h)
        ai2d_roi.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.roi_builder = ai2d_roi.build(self.roi_input_shape, self.roi_output_shape)
        self.roi_builder_cache_key = key
        return self.roi_builder

    def preprocess_det(self, img_chw):
        input_tensor = nn.from_numpy(img_chw)
        self.ai2d_det_builder.run(input_tensor, self.det_u8_out_tensor)
        det_u8 = self.det_u8_out_tensor.to_numpy()
        det_f32 = np.array(det_u8) / 255.0
        del input_tensor
        return nn.from_numpy(det_f32)

    def preprocess_roi(self, img_chw, x1, y1, x2, y2):
        x1 = clamp(int(x1), 0, self.ai_w - 1)
        y1 = clamp(int(y1), 0, self.ai_h - 1)
        x2 = clamp(int(x2), 0, self.ai_w - 1)
        y2 = clamp(int(y2), 0, self.ai_h - 1)

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 2 or crop_h < 2:
            return None

        input_tensor = nn.from_numpy(img_chw)
        self.get_roi_builder(x1, y1, crop_w, crop_h).run(input_tensor, self.roi_u8_tensor)
        roi_u8 = self.roi_u8_tensor.to_numpy()
        roi_f32 = np.array(roi_u8) / 255.0
        del input_tensor
        return nn.from_numpy(roi_f32)

    def run_kpu(self, kpu_obj, input_tensor):
        kpu_obj.set_input_tensor(0, input_tensor)
        kpu_obj.run()

        outputs = []
        for i in range(kpu_obj.outputs_size()):
            out_tensor = kpu_obj.get_output_tensor(i)
            outputs.append(out_tensor.to_numpy())
            del out_tensor
        return outputs

    def run_detect(self, img_chw):
        input_tensor = self.preprocess_det(img_chw)
        results = self.run_kpu(self.kpu_a, input_tensor)
        del input_tensor
        return results

    def run_state_cls(self, roi_tensor, part_id):
        outputs = self.run_kpu(self.kpu_b_nut if part_id == 0 else self.kpu_b_head, roi_tensor)
        logits = outputs[0][0]
        state_id, state_score = argmax_1d(logits)
        return state_id, float(state_score)

    def run_rust_cls(self, roi_tensor):
        outputs = self.run_kpu(self.kpu_c_rust, roi_tensor)
        logits = outputs[0][0]
        rust_id, rust_score = argmax_1d(logits)
        return rust_id, float(rust_score)

    def model_box_to_ai_box(self, x, y, w, h):
        x_scale = self.ai_w / A_INPUT_W
        y_scale = self.ai_h / A_INPUT_H

        x1 = int((x - 0.5 * w) * x_scale)
        y1 = int((y - 0.5 * h) * y_scale)
        x2 = int((x + 0.5 * w) * x_scale)
        y2 = int((y + 0.5 * h) * y_scale)

        return (
            clamp(x1, 0, self.ai_w - 1),
            clamp(y1, 0, self.ai_h - 1),
            clamp(x2, 0, self.ai_w - 1),
            clamp(y2, 0, self.ai_h - 1),
        )

    def ai_box_to_disp_box(self, x1, y1, x2, y2):
        sx1 = clamp(int(x1 * self.disp_w / self.ai_w), 0, self.disp_w - 1)
        sy1 = clamp(int(y1 * self.disp_h / self.ai_h), 0, self.disp_h - 1)
        sx2 = clamp(int(x2 * self.disp_w / self.ai_w), 0, self.disp_w - 1)
        sy2 = clamp(int(y2 * self.disp_h / self.ai_h), 0, self.disp_h - 1)
        return sx1, sy1, sx2, sy2

    def iou_box(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        xx1 = max(ax1, bx1)
        yy1 = max(ay1, by1)
        xx2 = min(ax2, bx2)
        yy2 = min(ay2, by2)

        w = max(0, xx2 - xx1 + 1)
        h = max(0, yy2 - yy1 + 1)
        inter = w * h

        area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
        area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
        denom = area_a + area_b - inter
        return 0.0 if denom <= 0 else inter / denom

    def find_cached_cls(self, bbox_ai):
        best_iou = 0.0
        best_item = None
        for item in self.last_cls_results:
            iou = self.iou_box(bbox_ai, item["bbox_ai"])
            if iou > best_iou:
                best_iou = iou
                best_item = item
        return best_item if best_iou >= 0.5 else None

    def postprocess_det(self, det_outputs):
        pred = det_outputs[0][0].transpose()
        boxes_ori = pred[:, 0:4]
        cls_scores = pred[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=-1)
        scores = np.max(cls_scores, axis=-1)

        boxes_ai = []
        keep_cls = []
        keep_scores = []

        for i in range(len(boxes_ori)):
            score = float(scores[i])
            if score < self.det_conf_thresh:
                continue

            x, y, w, h = boxes_ori[i][0], boxes_ori[i][1], boxes_ori[i][2], boxes_ori[i][3]
            x1, y1, x2, y2 = self.model_box_to_ai_box(x, y, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_ai.append([x1, y1, x2, y2])
            keep_cls.append(int(cls_ids[i]))
            keep_scores.append(score)

        if len(boxes_ai) == 0:
            return []

        keep = nms(boxes_ai, keep_scores, self.det_nms_thresh)
        dets = []
        for idx in keep:
            dets.append(
                {
                    "bbox_ai": boxes_ai[idx],
                    "part_id": int(keep_cls[idx]),
                    "part_name": A_LABELS[int(keep_cls[idx])],
                    "det_score": float(keep_scores[idx]),
                }
            )

        dets.sort(key=lambda x: x["det_score"], reverse=True)
        return dets[:MAX_BOXES_NUM]

    def run(self, img_chw):
        self.frame_id += 1
        det_outputs = self.run_detect(img_chw)
        dets = self.postprocess_det(det_outputs)
        final_results = []
        do_cls = self.frame_id % CLS_EVERY_N_FRAMES == 0

        for det in dets:
            x1, y1, x2, y2 = det["bbox_ai"]
            part_id = det["part_id"]

            state_id = -1
            state_name = "unknown"
            state_score = 0.0
            rust_id = -1
            rust_name = "skip"
            rust_score = 0.0

            if do_cls:
                roi_tensor = self.preprocess_roi(img_chw, x1, y1, x2, y2)
                if roi_tensor is None:
                    continue

                state_id, state_score = self.run_state_cls(roi_tensor, part_id)
                state_name = B_LABELS[state_id]

                need_rust = state_name == "tight" if ONLY_RUST_FOR_TIGHT else state_name != "missing"
                if need_rust:
                    rust_id, rust_score = self.run_rust_cls(roi_tensor)
                    rust_name = C_LABELS[rust_id]

                del roi_tensor
            else:
                cached = self.find_cached_cls(det["bbox_ai"])
                if cached is not None:
                    state_id = cached["state_id"]
                    state_name = cached["state_name"]
                    state_score = cached["state_score"]
                    rust_id = cached["rust_id"]
                    rust_name = cached["rust_name"]
                    rust_score = cached["rust_score"]

            dx1, dy1, dx2, dy2 = self.ai_box_to_disp_box(x1, y1, x2, y2)
            final_results.append(
                {
                    "bbox_ai": [x1, y1, x2, y2],
                    "bbox_disp": [dx1, dy1, dx2, dy2],
                    "bbox": [dx1, dy1, dx2 - dx1, dy2 - dy1],
                    "part_id": part_id,
                    "part_name": det["part_name"],
                    "det_score": det["det_score"],
                    "confidence": det["det_score"],
                    "state_id": state_id,
                    "state_name": state_name,
                    "state_score": state_score,
                    "rust_id": rust_id,
                    "rust_name": rust_name,
                    "rust_score": rust_score,
                }
            )

        if do_cls:
            self.last_cls_results = final_results

        if self.frame_id % GC_EVERY_N_FRAMES == 0:
            gc.collect()
        return final_results

    def draw_result(self, results, img):
        img.clear()
        for item in results:
            x1, y1, x2, y2 = item["bbox_disp"]
            color = A_COLORS[item["part_id"]]
            draw_w = int(x2 - x1)
            draw_h = int(y2 - y1)
            if draw_w <= 0 or draw_h <= 0:
                continue

            img.draw_rectangle(int(x1), int(y1), draw_w, draw_h, color=color, thickness=3)

            txt1 = "{} {}".format(item["part_name"], display_score_text(item["det_score"]))
            txt2 = "{} {:.3f}".format(item["state_name"], item["state_score"])

            if item["rust_name"] == "skip":
                txt3 = "rust: skip"
                txt3_color = COLOR_WHITE
            elif item["rust_name"] == "rusty":
                txt3 = "{} {:.3f}".format(item["rust_name"], item["rust_score"])
                txt3_color = COLOR_RED
            else:
                txt3 = "{} {:.3f}".format(item["rust_name"], item["rust_score"])
                txt3_color = COLOR_WHITE

            text_y = int(y1) - 54
            if text_y < 0:
                text_y = 0

            img.draw_string_advanced(int(x1), text_y, 20, txt1, color=color)
            img.draw_string_advanced(int(x1), text_y + 18, 20, txt2, color=COLOR_YELLOW)
            img.draw_string_advanced(int(x1), text_y + 36, 20, txt3, color=txt3_color)

    def deinit(self):
        for item in (
            "kpu_a",
            "kpu_b_nut",
            "kpu_b_head",
            "kpu_c_rust",
            "ai2d_det",
            "ai2d_det_builder",
            "det_u8_out_tensor",
            "roi_u8_tensor",
            "roi_builder",
        ):
            try:
                delattr(self, item)
            except Exception:
                pass
        nn.shrink_memory_pool()
        gc.collect()
        time.sleep_ms(50)
