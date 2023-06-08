import os

import supervision as sv
import torch
from autodistill.detection import DetectionTargetModel
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import \
    PPYoloEPostPredictionCallback

# necessary to prevent an error in the super_gradients library
import collections
collections.Iterable = collections.abc.Iterable

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HOME = os.path.expanduser("~")
MODEL_ARCH = "yolo_nas_l"
BATCH_SIZE = 8
CHECKPOINT_DIR = f"{HOME}/.cache/autodistill/yolonas/checkpoints"


class YOLONAS(DetectionTargetModel):
    model_name: str

    def __init__(self, model_name):
        self.model_name = model_name
        self.yolonas = None
        self.classes = None
        self.dataset_location = None

    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        if self.yolonas is None:
            raise ("A YOLO-NAS model has not yet been loaded.")

        result = list(self.yolonas.predict(input, conf=confidence))[0]

        return sv.Detections.from_yolo_nas(result)

    def train(self, dataset_location: str, epochs: int = 25):
        data = sv.DetectionDataset.from_yolo(annotations_directory_path=dataset_location, data_yaml_path=dataset_location.strip("/") + "/data.yaml", images_directory_path=dataset_location)
        classes = data.classes
        self.classes = classes
        self.dataset_location = dataset_location

        trainer = Trainer(experiment_name=self.model_name, ckpt_root_dir=CHECKPOINT_DIR)

        dataset_params = {
            "data_dir": dataset_location,
            "train_images_dir": "train/images",
            "train_labels_dir": "train/labels",
            "val_images_dir": "valid/images",
            "val_labels_dir": "valid/labels",
            "test_images_dir": "test/images",
            "test_labels_dir": "test/labels",
            "classes": classes,
        }

        train_data = coco_detection_yolo_format_train(
            dataset_params={
                "data_dir": dataset_params["data_dir"],
                "images_dir": dataset_params["train_images_dir"],
                "labels_dir": dataset_params["train_labels_dir"],
                "classes": dataset_params["classes"],
            },
            dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 2},
        )

        self.dataset_params = dataset_params

        val_data = coco_detection_yolo_format_val(
            dataset_params={
                "data_dir": dataset_params["data_dir"],
                "images_dir": dataset_params["val_images_dir"],
                "labels_dir": dataset_params["val_labels_dir"],
                "classes": dataset_params["classes"],
            },
            dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 2},
        )

        model = models.get(
            MODEL_ARCH,
            num_classes=len(dataset_params["classes"]),
            pretrained_weights="coco",
        )

        train_params = {
            "silent_mode": False,
            "average_best_models": True,
            "warmup_mode": "linear_epoch_step",
            "warmup_initial_lr": 1e-6,
            "lr_warmup_epochs": 3,
            "initial_lr": 5e-4,
            "lr_mode": "cosine",
            "cosine_final_lr_ratio": 0.1,
            "optimizer": "Adam",
            "optimizer_params": {"weight_decay": 0.0001},
            "zero_weight_decay_on_bias_and_bn": True,
            "ema": True,
            "ema_params": {"decay": 0.9, "decay_type": "threshold"},
            "max_epochs": epochs,
            "mixed_precision": True if DEVICE == "cuda" else False,
            "loss": PPYoloELoss(
                use_static_assigner=False,
                num_classes=len(dataset_params["classes"]),
                reg_max=16,
            ),
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=len(dataset_params["classes"]),
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.01,
                        nms_top_k=1000,
                        max_predictions=300,
                        nms_threshold=0.7,
                    ),
                )
            ],
            "metric_to_watch": "mAP@0.50",
        }

        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_data,
            valid_loader=val_data,
        )

        self.yolonas = models.get(
            MODEL_ARCH,
            num_classes=len(self.dataset_params["classes"]),
            checkpoint_path=f"{CHECKPOINT_DIR}/{self.model_name}/average_model.pth",
        ).to(DEVICE)
