import sys
import cv2

from loggers import *

import signal
import yaml
import time
import multiprocessing as mp
import threading as thr
import base64, json
import datetime
from det_paddle import OCR

from config_router2 import ConfigRouter

logger = logging.getLogger(__name__)

STOP = mp.Event()


def _loop(proc_configs, interval=2.0, max_restarts=3):
    while not STOP.is_set():
        for cfg in proc_configs:
            p = cfg['proc']
            if p is None or p.is_alive(): continue

            logger.warning(f"⚠️ Процесс '{cfg['name']}' упал (exitcode={p.exitcode})")
            try:
                p.join(timeout=1)
            except:
                pass

            if cfg['restarts'] < max_restarts:
                cfg['restarts'] += 1
                #logger.info(f"🔄 Перезапуск '{cfg['name']}' ({cfg['restarts']}/{max_restarts})")
                cfg['proc'] = mp.Process(target=cfg['target'], args=cfg['args'],
                                         daemon=cfg['daemon'], name=cfg['name'])
                cfg['proc'].start()
            else:
                #logger.critical(f"💀 Лимит рестартов для '{cfg['name']}'. Остановка пайплайна.")
                STOP.set()
        STOP.wait(interval)  # ждёт сигнал или таймаут

def setup_signals(proc_configs):
    def handler(sig, frame):
        logger.warning(f"🛑 Сигнал {sig}. Корректное завершение...")
        STOP.set()
        for cfg in proc_configs:
            if cfg['proc'] and cfg['proc'].is_alive(): cfg['proc'].terminate()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

def start_watchdog(procs, interval=2.0, max_restarts=3):
    """
    proc_configs: список словарей вида:
    [{'proc': p, 'target': func, 'args': (), 'daemon': True, 'name': 'MyProc', 'restarts': 0}, ...]
    """

    thr.Thread(target=_loop, args=(procs, interval, max_restarts), daemon=True, name="Watchdog").start()



def set_cam_param(cap, set_w, set_h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, set_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, set_h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    #print(cap.get(cv2.CAP_PROP_FPS), cap.getBackendName())

def encode_frame_for_frontend(frame, quality=85, format='.jpg'):
    _, buffer = cv2.imencode(format, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


    return timestamp, json.dumps({
        "timestamp": timestamp,
        "frame_b64": base64.b64encode(buffer).decode('utf-8')
    })



def process_video(q_vid:mp.Queue, path_video, realtime_queue: bool = False):



    cap = cv2.VideoCapture(path_video)


    if not cap.isOpened():
        return

    c = 0

    while True:

        try:
            flag, image = cap.read()
            h, w, _ = image.shape
            if flag:
                c = 0
                if realtime_queue:
                    if q_vid.empty():
                        time_stmp, frame_dump = encode_frame_for_frontend(image)
                        q_vid.put([image, time_stmp, flag])
                else:
                    time_stmp, frame_dump = encode_frame_for_frontend(image)
                    q_vid.put([image, time_stmp, flag])




        except Exception as e:
            logger.error(f"Errr_connect_cam_or_video: {e.args}")
            cap.release()
            cv2.destroyAllWindows()
            q_vid.put([None, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], False])
            time.sleep(3)

            cap = cv2.VideoCapture(path_video)

            c += 1
            logger.info(f"Reconn_cam/video: {cap.isOpened()}, count try connect: {c}")
            continue

    cap.release()
    logger.info("🎬 Video processor stopped")



if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    from yolo_det_class import *
    # Инициализация конфига
    CONFIG_PATH = "config_pipeline2.yaml"
    cfg_router = ConfigRouter(CONFIG_PATH)

    log_level = cfg_router.get('system.log_level', 'INFO')
    #logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))


    video_source = cfg_router.get_video_source()
    realtime_mode = cfg_router.is_realtime_queue()

    #logger.info(f"📹 Source: {video_source}")
    #logger.info(f"📮 Queue mode: {'REALTIME' if realtime_mode else 'ACCUMULATE'}")



    q_to_yolo = mp.Queue(maxsize=1)
    q_to_ocr = mp.Queue(maxsize=1)


    prc_video = mp.Process(target=process_video, args=(q_to_yolo, video_source, realtime_mode,), daemon=True, name="VideoReader")
    prc_video.start()

    cls_det = Yolo_detection(q_to_yolo, q_to_ocr)

    yolo_obj_mapping = {
        'model_path': 'path_yolo_obj_det',
        'img_size': 'size_inp_layers_yolo_obj_det',
        'conf': 'conf_model_yolo_obj_det',
        'device': 'device_yolo_obj_det',
        'half': 'half_flag_yolo_obj_det',
        'verbose': 'verbose_yolo_obj_det'
    }
    cls_det.set_path_yolo_obj_det(cfg_router.get_model_path_trt('obj_det'))
    cls_det.set_path_yolo_plate(cfg_router.get_model_path_trt('plate_det'))
    cfg_router.apply_to_object(cls_det, 'yolo.obj_det', yolo_obj_mapping)

    yolo_plate_mapping = {
        'model_path': 'path_yolo_plate',
        'img_size': 'size_inp_layers_yolo_plate',
        'conf': 'conf_model_yolo_plate',
        'device': 'device_yolo_plate',
        'half': 'half_flag_yolo_plate',
        'verbose': 'verbose_yolo_plate'
    }
    cfg_router.apply_to_object(cls_det, 'yolo.plate_det', yolo_plate_mapping)

    roi_coords = cfg_router.get_roi()
    if roi_coords:
        cls_det.update_roi(roi_coords)
        logger.info(f"🎯 ROI applied: {roi_coords}")

    cfg_router.apply_render_to_yolo(cls_det)  # ← добавит time_delay
    prc_det = cls_det.run_process()


    ocr_obj = OCR(q_to_ocr)

    ocr_mapping = {
        'cpu_threads': 'cpu_threads',
        'use_doc_orientation_classify': 'use_doc_orientation_classify',
        'use_doc_unwarping': 'use_doc_unwarping',
        'use_textline_orientation': 'use_textline_orientation',
        'render_timeout': 'render_timeout'
    }


    cfg_router.apply_to_object(ocr_obj, 'ocr', ocr_mapping)
    cfg_router.apply_render_to_ocr(ocr_obj)  # ← добавит render_timeout

    prc_ocr = ocr_obj.run_process()

    proc_configs = [
        {'proc': prc_video, 'target': process_video,
         'args': (q_to_yolo, video_source, realtime_mode),
         'daemon': True, 'name': 'VideoReader', 'restarts': 0},
        {'proc': prc_det, 'target': cls_det.main_process, 'args': (),
         'daemon': True, 'name': 'YOLODetector', 'restarts': 0},
        {'proc': prc_ocr, 'target': ocr_obj.main_proc, 'args': (q_to_ocr,),
         'daemon': True, 'name': 'OCR', 'restarts': 0}
    ]


    setup_signals(proc_configs)
    start_watchdog(proc_configs,
                   interval=cfg_router.get('system.watchdog_interval', 2.0),
                   max_restarts=cfg_router.get('system.max_restarts', 3))

    logger.info("🚀 Pipeline started. Press Ctrl+C to stop.")

    try:
        while not STOP.is_set():
            if cfg_router.reload():
                new_roi = cfg_router.get_roi()
                if new_roi and hasattr(cls_det, 'update_roi'):
                    cls_det.update_roi(new_roi)
                    logger.info(f"🔄 ROI updated: {new_roi}")
            time.sleep(10)
    except KeyboardInterrupt:
        pass
        #logger.info("⌨️ Interrupted")
    finally:
        STOP.set()
        cv2.destroyAllWindows()
        #logger.info("🔚 Stopped")