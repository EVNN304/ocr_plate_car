from ultralytics import YOLO
import torch




class Converter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.dict_par = {"format" :'engine',
            "device":0,
            "half":True,
            "int8": False,
            "dynamic":True,
            "simplify":True,
            "workspace":1,
            "imgsz":(288, 288),
            "batch":64,
            "nms":False,
            "verbose":True}

    def set_inp_path_model(self, val:str):
        self.model_path = val


    def set_export_param(self, key, value):
        self.dict_par[key] = value

    def start_to_convert(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("=" * 60)
            print(f"Экспорт в TensorRT Engine (batch до {self.dict_par['batch']})")
            print("=" * 60)
            model = YOLO(self.model_path)
            model.export(**self.dict_par)
            print(f"\n✅ Engine создан: {self.model_path[:-2]+'engine'} успешно!🔥")
        except Exception as e:
            print(f"❌ Errrr_export: {e.args}")



if __name__ == '__main__':
    #MODEL_PT = '/home/usr/Рабочий стол/weights_yolo26/drone_iter_3_m/train23/weights/best.pt'
    MODEL_PT = f"/home/usr/Рабочий стол/weights_yolo26/IR_YOLO26m/train32/weights/best_IR_2.pt"
    SLICE_SIZE = 288
    MAX_BATCH = 64


    cls = Converter(MODEL_PT)
    cls.set_export_param("imgsz", (SLICE_SIZE, SLICE_SIZE))
    cls.set_export_param("batch", MAX_BATCH)
    cls.set_export_param("workspace", 1.25)

    cls.start_to_convert()

