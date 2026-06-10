import cv2 as cv
import multiprocessing as mp
import threading as th
import base64, json
import time
from paddleocr import PaddleOCR



class OCR:
    def __init__(self, q_ocr:mp.Queue):
        self.q_ocr = q_ocr
        self.cpu_threads = 16
        self.use_doc_orientation_classify = False
        self.use_doc_unwarping = False
        self.use_textline_orientation = False
        self.render_timeout = 1
        self.message = {"schema_version": "1",
                   "event_id":"550e8400-e29b-41d4-a716-446655440000",
                   "lifecycle":"lost", "track_id": "track-42",
                   "occurred_at":"2026-05-21T12:00:00.000Z",
                   "source":{"site_id":"00000000-0000-0000-0000-000000000002",
                             "camera_id":"cam-main", "lane_id":"lane-in"},
                   "plate":{"text": "A000AA72", "confidence": 0.92, "country": "RU", "bbox": [100, 200, 300, 400]},
                   "lists":{"matched": "unknown", "frame_stamp":"2020-05-21T12:00:00.000Z", "area_detect":[100, 200, 300, 400]},
                   "payload":{"bbox": [100, 200, 300, 400], "vehicle_type": "car", "timestamp": "2020-05-21T12:00:00.000Z", "confidence":0.77}}

        self.flag_send = False



    def set_cpu_threads(self, val:int):
        self.cpu_threads = val


    def set_use_doc_orientation_classify(self, val:bool):
        self.use_doc_orientation_classify = val

    def set_use_doc_unwarping(self, val:bool):
        self.use_doc_unwarping = val

    def set_use_textline_orientation(self, val:bool):
        self.use_textline_orientation = val


    def set_render_timeout(self, val:int):
        self.render_timeout = val


    def load_model_ocr(self):
        ocr = PaddleOCR(cpu_threads=self.cpu_threads,
                        use_doc_orientation_classify=self.use_doc_orientation_classify,
                        use_doc_unwarping=self.use_doc_unwarping,
                        use_textline_orientation=self.use_textline_orientation)

        return ocr



    def run_process(self, daemon=True):
        proc = mp.Process(target=self.main_proc, args=(), daemon=daemon)
        proc.start()
        return proc


    def update_message(self, lst_obj, text_plate):


        self.message["plate"]["text"] = text_plate
        self.message["plate"]["bbox"] = lst_obj[0]["plate_box"]
        self.message["plate"]["confidence"] = lst_obj[0]["plate_conf"]


        self.message["lists"]["frame_stamp"] = lst_obj[0]["time_stamp_frame"]
        self.message["lists"]["area_detect"] = lst_obj[0]["area_det"]



        self.message["payload"]["bbox"] = lst_obj[0]["car_box"]
        self.message["payload"]["vehicle_type"] = lst_obj[0]["vehicle_type"]
        self.message["payload"]["confidence"] = lst_obj[0]["conf_vehicle"]
        self.message["payload"]["timestamp"] = lst_obj[0]["detection_timestamp"]




    def run_thread(self, daemon=True):
        send_thr = th.Thread(target=self.send_message, args=(), daemon=daemon)
        send_thr.start()


    def send_message(self):

        while True:
            if self.flag_send:
                print(f"SEND_MESSAGE: {self.message}")
            else:
                print("END")
            time.sleep(0.45)





    def main_proc(self):

        ocr = self.load_model_ocr()
        self.run_thread()
        while True:
            if not self.q_ocr.empty():
                images, dets, flag = self.q_ocr.get()
                try:

                    if images is not None:
                        for k in dets:
                            car_box = k.get("car_box")
                            crop = images[int(k["plate_box"][1]):int(k["plate_box"][3]), int(k["plate_box"][0]):int(k["plate_box"][2])]

                            result = ocr.predict(input=crop)

                            texts = result[0].get('rec_texts', [])
                            plate_text = " ".join(texts) if texts else "NO PLATE"
                            self.flag_send = True

                            self.update_message(dets, plate_text)
                            x1_c, y1_c, x2_c, y2_c = map(int, car_box)

                            cv.rectangle(images, (x1_c, y1_c), (x2_c, y2_c), (255, 0, 0), 2)

                            (tw, th), baseline = cv.getTextSize(plate_text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            margin = 10
                            text_pos = (x2_c - tw - margin, y2_c - margin)

                            # Подложка для читаемости
                            cv.rectangle(images, (text_pos[0] - 5, text_pos[1] - th - 5),
                                         (text_pos[0] + tw + 5, text_pos[1] + baseline + 5),
                                         (255, 255, 255), -1)
                            # Пишем текст
                            cv.putText(images, plate_text, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        cv.imshow("Result", images)
                        cv.waitKey(self.render_timeout)

                    if dets == [] or (images is None) or flag == False:
                        self.flag_send = False
                except Exception as e:
                    print("FFF", e.args)

