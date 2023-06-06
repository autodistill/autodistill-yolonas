import supervision as sv
import torch
import os
from autodistill.detection import DetectionTargetModel

from super_gradients.training import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

HOME = os.path.expanduser("~")
MODEL_ARCH = 'yolo_nas_l'
BATCH_SIZE = 8
MAX_EPOCHS = 25
CHECKPOINT_DIR = f'{HOME}/.cache/autodistill/yolonas/checkpoints'

trainer = Trainer(experiment_name="yolonas", ckpt_root_dir=CHECKPOINT_DIR)

class YOLONAS(DetectionTargetModel):
    def __init__(self, model_name):
        self.yolo = trainer.load_model(MODEL_ARCH, ckpt_dir=CHECKPOINT_DIR)

        self.yolo.conf = 0.25  # NMS confidence threshold
        self.yolo.iou = 0.45  # NMS IoU threshold
        self.yolo.agnostic = False  # NMS class-agnostic
        self.yolo.multi_label = False  # NMS multiple labels per box
        self.yolo.max_det = 1000  # maximum number of detections per image

    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        predictions = self.yolo(
            source=input,
            imgsz=640,
            conf_thres=confidence,
            iou_thres=0.5,
            max_det=1000,
            device=device,
        )

        return predictions

    def train(self, dataset_yaml, epochs=300, image_size=640):
        train.run(data=dataset_yaml, epochs=epochs, imgsz=image_size)
