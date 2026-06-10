import time
import sys
import cv2
import torch
import datetime
import multiprocessing as mp

from ultralytics import YOLO
from loggers import *



class Yolo_detection:
    def __init__(self, q_to_yolo:mp.Queue, q_to_ocr:mp.Queue):

        self.q_to_yolo = q_to_yolo
        self.q_to_ocr = q_to_ocr
        self.pth_yolo_obj_det = f"yolo26x.pt"
        self.pth_yolo_plate = f"/home/usr/Downloads/best.pt"
        
        self.imgsize_det_obj = (640, 640)
        self.half_flag_det_obj = True
        self.device_det_obj = 0
        self.verbose_det_obj = True
        self.clases_names = {}
        self.conf_det_obj = 0.55

        self.imgsize_det_plate = (640, 640)
        self.half_flag_det_plate = True
        self.device_det_plate = 0
        self.verbose_det_plate = True
        self.conf_det_plate = 0.55

        self.allowed_vehicles = {'car', 'motorcycle', 'truck'}
        self.time_delay = 1
        self.roi_enabled = False
        self.roi_coords = None

        #self.logger = logging.getLogger(__name__)

    def set_verbose_yolo_obj_det(self, val:bool):
        self.verbose_det_obj = val


    def set_time_delay(self, val:int):
        self.time_delay = val

    def set_path_yolo_obj_det(self, val:str):
        self.pth_yolo_obj_det = val

    def set_path_yolo_plate(self, val:str):
        self.pth_yolo_plate = val

    def set_size_inp_layers_yolo_obj_det(self, val:tuple):
        self.imgsize_det_obj = val

    def set_half_flag_yolo_obj_det(self, val:bool):
        self.half_flag_det_obj = val


    def set_allowed_vehicles(self, val:set):
        self.allowed_vehicles = val


    def set_device_yolo_obj_det(self, val):
        self.device_det_obj = val

    def set_classes_names(self, val:dict):
        self.clases_names = val

    def set_conf_model_yolo_obj_det(self, val:float):
        self.conf_det_obj = val

    def set_verbose_yolo_plate(self, val:bool):
        self.verbose_det_plate = val

    def set_size_inp_layers_yolo_plate(self, val:tuple):
        self.imgsize_det_plate = val

    def set_half_flag_yolo_plate(self, val:bool):
        self.half_flag_det_plate = val

    def set_device_yolo_plate(self, val):
        self.device_det_plate = val

    def set_conf_model_yolo_plate(self, val:float):
        self.conf_det_plate = val

    def run_process(self, daemon=True):

        proc = mp.Process(target=self.main_process, args=(), daemon=daemon)
        proc.start()

        return proc

    def load_model_yolo_det(self):
        try:
            return YOLO(self.pth_yolo_obj_det)
        except Exception as e:
            self.logger.error(f"Error_load_model_yolo_obj_det: {e.args}")
            return None


    def load_model_yolo_plate(self):
        try:
            return YOLO(self.pth_yolo_plate)
        except Exception as e:
            self.logger.error(f"Error_load_model_yolo_plate: {e.args}")
            return None


    def update_roi(self, coords: list):
        self.roi_enabled = True
        self.roi_coords = [int(c) for c in coords]


    def main_process(self):
        model_obj_det = self.load_model_yolo_det()
        self.set_classes_names(model_obj_det.names)
        model_det_plate = self.load_model_yolo_plate()
        frame_count = 0



        while True:
            if not self.q_to_yolo.empty():
                frame, time_stmp, flag = self.q_to_yolo.get()
                if frame is not None:

                    draw_frame = frame.copy()
                    cars_data = []
                    frame_count += 1

                    if self.roi_enabled and self.roi_coords:
                        rx1, ry1, rx2, ry2 = self.roi_coords
                        cv2.rectangle(draw_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                        cv2.putText(draw_frame, "ROI ZONE", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    res_cars = model_obj_det(
                        frame, verbose=self.verbose_det_obj, device=self.device_det_obj,
                        conf=self.conf_det_obj, imgsz=self.imgsize_det_obj, half=self.half_flag_det_obj
                    )

                    for box in res_cars[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.clases_names[cls_id]

                        if cls_name not in self.allowed_vehicles:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])

                        if self.roi_enabled and self.roi_coords:
                            rx1, ry1, rx2, ry2 = self.roi_coords
                            if x1 < rx1 or y1 < ry1 or x2 > rx2 or y2 > ry2:
                                continue

                        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(draw_frame, f"{cls_name.capitalize()} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        car_crop = frame[y1:y2, x1:x2]
                        if car_crop.size == 0:
                            continue

                        res_pl = model_det_plate(
                            car_crop, verbose=self.verbose_det_plate, device=self.device_det_plate,
                            conf=self.conf_det_plate, imgsz=self.imgsize_det_plate, half=self.half_flag_det_plate
                        )

                        for p_box in res_pl[0].boxes:
                            px1, py1, px2, py2 = map(int, p_box.xyxy[0].tolist())
                            p_conf = float(p_box.conf[0])

                            gx1, gy1 = x1 + px1, y1 + py1
                            gx2, gy2 = x1 + px2, y1 + py2

                            cv2.rectangle(draw_frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                            cv2.putText(draw_frame, f"Plate {p_conf:.2f}", (gx1, gy1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            cars_data.append({
                                "car_box": [x1, y1, x2, y2],
                                "plate_box": [gx1, gy1, gx2, gy2],
                                "plate_conf": p_conf,
                                "time_stamp_frame": time_stmp,
                                "vehicle_type": cls_name,
                                "conf_vehicle": conf,
                                "detection_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                "area_det": self.roi_coords
                            })

                    cv2.imshow("Detections", draw_frame)
                    if cv2.waitKey(self.time_delay) & 0xFF == 27:
                        break

                if self.q_to_ocr.empty():
                    self.q_to_ocr.put([frame, cars_data, flag])


