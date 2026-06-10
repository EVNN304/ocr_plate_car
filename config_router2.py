import yaml
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from convert_weights_TRT import Converter

logger = logging.getLogger(__name__)


class ConfigRouter:
    def __init__(self, config_path: str, validate: bool = True):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._last_mtime: float = 0
        self.validate = validate

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        self.reload()

    def reload(self, force: bool = False) -> bool:
        try:
            current_mtime = self.config_path.stat().st_mtime
            if force or current_mtime > self._last_mtime:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
                self._last_mtime = current_mtime
                logger.info(f"✅ Config reloaded: {self.config_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Config reload failed: {e}")
            return False

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get(self, path: str, default: Any = None) -> Any:
        try:
            value = self._config
            for key in path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_converted_weights_path(self, original_path: str) -> str:
        """
        Возвращает путь к весам: либо оригинальный .pt, либо сконвертированный .engine.
        Конвертация происходит только если:
        - включена в конфиге (tensorrt.enabled: true)
        - исходный файл имеет расширение .pt
        - engine ещё не существует рядом
        """
        # ✅ Исправлено: self._config вместо self.cfg
        trt_cfg = self._config.get("tensorrt", {})

        # 1. Если конвертация выключена → возвращаем как есть
        if not trt_cfg.get("enabled", False):
            return original_path

        # 2. Если уже .engine → не трогаем
        if original_path.lower().endswith(".engine"):
            logger.info(f"✅ TRT: Already engine format: {original_path}")
            return original_path

        # 3. Если engine уже есть рядом → кэш
        engine_path = os.path.splitext(original_path)[0] + ".engine"
        if os.path.exists(engine_path):
            logger.info(f"✅ TRT: Engine exists, skipping conversion: {engine_path}")
            return engine_path

        # 4. Запуск конвертации
        logger.info(f"🔧 TRT: Converting {original_path} → {engine_path} ...")
        try:
            converter = Converter(original_path)

            # Пробрасываем ВСЕ параметры из конфига — ТОЧНО как в рабочем проекте
            converter.set_export_param("format", trt_cfg.get("format", "engine"))
            converter.set_export_param("device", trt_cfg.get("device", 0))
            converter.set_export_param("half", trt_cfg.get("half", True))
            converter.set_export_param("int8", trt_cfg.get("int8", False))
            converter.set_export_param("dynamic", trt_cfg.get("dynamic", True))
            converter.set_export_param("simplify", trt_cfg.get("simplify", True))
            converter.set_export_param("workspace", trt_cfg.get("workspace", 1.0))
            converter.set_export_param("imgsz", tuple(trt_cfg.get("imgsz", [288, 288])))
            converter.set_export_param("batch", trt_cfg.get("batch", 16))
            converter.set_export_param("nms", trt_cfg.get("nms", False))
            converter.set_export_param("verbose", trt_cfg.get("verbose", True))

            converter.start_to_convert()

            if os.path.exists(engine_path):
                logger.info(f"✅ TRT: Conversion successful. Using {engine_path}")
                return engine_path
            else:
                logger.warning(f"⚠️ TRT: Conversion finished but engine not found. Fallback to {original_path}")
                return original_path

        except Exception as e:
            logger.error(f"❌ TRT: Conversion failed: {e}", exc_info=True)
            return original_path

    def get_model_path_trt(self, model_key: str) -> str:
        """
        Возвращает путь к модели с учётом TensorRT.
        :param model_key: 'obj_det' или 'plate_det'
        """
        # ✅ Используем self.get() для совместимости с reload()
        original_path = self.get(f'yolo.{model_key}.model_path', '')
        return self.get_converted_weights_path(original_path)

    def get_render_params(self) -> dict:
        """Получить параметры отрисовки"""
        return {
            'time_delay': self.get('render.time_delay', 1),
            'render_timeout': self.get('render.render_timeout', 1)
        }

    def apply_render_to_yolo(self, yolo_obj):
        """Применить render-параметры к Yolo_detection"""
        params = self.get_render_params()
        if hasattr(yolo_obj, 'set_time_delay'):
            yolo_obj.set_time_delay(params['time_delay'])
        logger.debug(f"🎨 Render params applied to YOLO: {params}")

    def apply_render_to_ocr(self, ocr_obj):
        """Применить render-параметры к OCR"""
        params = self.get_render_params()
        if hasattr(ocr_obj, 'set_render_timeout'):
            ocr_obj.set_render_timeout(params['render_timeout'])
        logger.debug(f"🎨 Render params applied to OCR: {params}")


    def apply_to_object(self, obj: Any, prefix: str, mapping: Dict[str, str], strict: bool = False):
        section = self.get(prefix, {})
        if not section:
            logger.warning(f"⚠️ Empty config section: {prefix}")
            return

        for cfg_key, setter_suffix in mapping.items():
            if cfg_key not in section:
                continue

            value = section[cfg_key]
            setter_name = f"set_{setter_suffix}"

            if hasattr(obj, setter_name) and callable(getattr(obj, setter_name)):
                try:
                    if self.validate and setter_suffix in ['conf_model_yolo_obj_det', 'conf_model_yolo_plate']:
                        if not (0.0 <= float(value) <= 1.0):
                            logger.warning(f"⚠️ Conf value out of range [0,1]: {value}")
                            continue

                    getattr(obj, setter_name)(value)
                    logger.debug(f"🔧 Applied {prefix}.{cfg_key} → {setter_name}({value})")
                except Exception as e:
                    logger.error(f"❌ Failed to apply {cfg_key}: {e}")
                    if strict:
                        raise
            else:
                msg = f"⚠️ Setter '{setter_name}' not found in {type(obj).__name__}"
                if strict:
                    raise AttributeError(msg)
                logger.warning(msg)

    def get_roi(self) -> Optional[list]:
        roi = self.get('roi', {})
        if roi.get('enabled') and 'coordinates' in roi:
            coords = roi['coordinates']
            if len(coords) == 4 and all(isinstance(c, (int, float)) for c in coords):
                return [int(c) for c in coords]
        return None

    def get_video_source(self):
        return self.get('video.source', 0)

    def is_realtime_queue(self) -> bool:
        return self.get('queue.realtime', True)

    def get_ocr_params(self) -> Dict[str, Any]:
        return self.get('ocr', {})