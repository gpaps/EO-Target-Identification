import os, json, argparse, warnings, logging
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.data import transforms as T
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.data import MetadataCatalog
from register_ships_dataset import register_datasets
from torch.utils.tensorboard import SummaryWriter

logging.getLogger("detectron2").setLevel(logging.WARN)
warnings.simplefilter(action='ignore', category=FutureWarning)
Image.MAX_IMAGE_PIXELS = 100_000_000

register_datasets()


class ShipSARMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.cfg = cfg
        self.is_train = is_train
        self.augmentations = T.AugmentationList([
            T.ResizeShortestEdge(short_edge_length=(640, 768, 896), sample_style="choice"),
            T.RandomBrightness(0.75, 1.25),
            T.RandomContrast(0.7, 1.3),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.2, horizontal=False, vertical=True),
            T.RandomRotation(angle=[-10, 10]),
        ])

    def __call__(self, dataset_dict):
        if not self.is_train and not getattr(self.cfg, "KEEP_GT_IN_VAL", False):
            dataset_dict.pop("annotations", None)
        return super().__call__(dataset_dict)


class MiniAirEvaluator:
    def __init__(self, df, output_dir, class_names):
        self.df = df.copy()
        self.output_dir = output_dir
        self.class_names = class_names
        self.bg_idx = len(class_names)

    def plot_canonical_confusion_matrix(self):
        y_true, y_pred = [], []
        df = self.df

        for img in df["image"].unique():
            df_img = df[df["image"] == img]
            gt = df_img[df_img["gt_class"] != -1]
            pred = df_img[df_img["pred_class"] != -1]
            matched_gt = set()
            matched_pred = set()

            for pi, prow in pred.iterrows():
                matches = gt[(gt["iou"] >= 0.5) & (~gt.index.isin(matched_gt))]
                if matches.empty:
                    y_true.append(self.bg_idx)
                    y_pred.append(prow["pred_class"])
                    continue
                best_gt = matches.loc[matches["iou"].idxmax()]
                matched_gt.add(best_gt.name)
                matched_pred.add(pi)
                y_true.append(int(best_gt["gt_class"]))
                y_pred.append(int(prow["pred_class"]))

            for gi in gt.index.difference(matched_gt):
                y_true.append(int(gt.loc[gi]["gt_class"]))
                y_pred.append(self.bg_idx)

            for pi in pred.index.difference(matched_pred):
                y_true.append(self.bg_idx)
                y_pred.append(int(pred.loc[pi]["pred_class"]))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.bg_idx + 1)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names + ["background"])
        disp.plot(xticks_rotation=45, cmap="Purples", values_format="d")
        plt.title("Canonical Confusion Matrix (Val Eval)")
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "confusion_matrix_val.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        log_image_to_tensorboard(fig_path, "Val/Confusion_Matrix", os.path.join(self.output_dir, "tensorboard"))

    def plot_precision_recall_f1_table(self):
        df = self.df
        cm_path = os.path.join(self.output_dir, "confusion_matrix_val.csv")
        if not os.path.exists(cm_path):
            return

        df_cm = pd.read_csv(cm_path, index_col=0)
        results = []
        for i in range(len(self.class_names)):
            TP = df_cm.iloc[i, i]
            FP = df_cm.iloc[-1, i]
            FN = df_cm.iloc[i, -1]
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            results.append({"Class": self.class_names[i], "Precision": precision, "Recall": recall, "F1": f1})

        df_results = pd.DataFrame(results)
        plt.figure(figsize=(10, len(df_results) * 0.6))
        sns.heatmap(df_results.set_index("Class"), annot=True, fmt=".2f", cmap="Blues", cbar=False)
        plt.title("Precision / Recall / F1 on Validation")
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "val_metrics_table.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        log_image_to_tensorboard(fig_path, "Val/F1_Precision_Table", os.path.join(self.output_dir, "tensorboard"))


class EvalHook(HookBase):
    def __init__(self, eval_period, model, val_loader, output_dir, class_names, patience):
        self._period = eval_period
        self.model = model
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.class_names = class_names
        self.patience = patience
        self.best_ap = -1
        self.counter = 0

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if next_iter % self._period == 0 or is_final:
            print(f"[EvalHook] Eval results at iter {next_iter}: {results}")
            results = inference_on_dataset(
                self.model,
                self.val_loader,
                COCOEvaluator(self.trainer.cfg.DATASETS.TEST[0], self.trainer.cfg, False, self.output_dir)

            )

            current_ap = results.get("bbox", {}).get("AP", -1)
            print(f"[EvalHook] Iter {next_iter} AP: {current_ap:.2f}")

            # Mid-eval visuals/logs
            pred_log = os.path.join(self.output_dir, "prediction_log.csv")
            if os.path.exists(pred_log):
                df = pd.read_csv(pred_log)
                evaluator = MiniAirEvaluator(df, self.output_dir, self.class_names)
                evaluator.plot_canonical_confusion_matrix()
                evaluator.plot_precision_recall_f1_table()

            # Early stopping logic
            if current_ap > self.best_ap:
                self.best_ap = current_ap
                self.counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "model_best.pth"))
            else:
                self.counter += 1
                print(f"[EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print(" Early stopping triggered.")
                self.trainer.storage.put_scalars(early_stop_iter=next_iter)
                raise Exception("EARLY_STOP")


class AugmentedTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._writers = self.build_writers()

    def build_writers(self):
        return [
            CommonMetricPrinter(self.cfg.SOLVER.MAX_ITER),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(os.path.join(self.cfg.OUTPUT_DIR, "tensorboard"))
        ]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
        class_names = MetadataCatalog.get(self.cfg.DATASETS.TEST[0]).thing_classes
        hooks.insert(-1,
                     EvalHook(
                         self.cfg.TEST.EVAL_PERIOD,
                         self.model,
                         val_loader,
                         self.cfg.OUTPUT_DIR,
                         class_names,
                         self.cfg.EARLY_STOP.PATIENCE
                     )
                     )
        return hooks


def setup_and_train(output_dir, num_classes, lr=0.0001, roi_batch=512, score=.5, nms=.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("Ships_SAR_train",)
    cfg.DATASETS.TEST = ("Ships_SAR_val",)
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # Core Solver Params
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = 20000  # 40000
    cfg.SOLVER.STEPS = (14000, 18000)  # (24000, 32000)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WEIGHT_DECAY = 0.0001  # 0.00005
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000
    cfg.TEST.EVAL_PERIOD = 2000
    cfg.SOLVER.AMP.ENABLED = True
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    print(f"🧪 FILTER_EMPTY_ANNOTATIONS: {cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS}")

    # Input
    cfg.INPUT.MIN_SIZE_TRAIN = [640, 768, 896]
    cfg.INPUT.MAX_SIZE_TRAIN = 1920
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    # ROI HEAD
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_batch
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score  # was .3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms

    cfg.EARLY_STOP = CN()
    cfg.EARLY_STOP.PATIENCE = 3  # We can override this in argparse later if needed

    cfg.OUTPUT_DIR = output_dir
    cfg.KEEP_GT_IN_VAL = True
    os.makedirs(output_dir, exist_ok=True)

    logText(cfg, cfg.OUTPUT_DIR)

    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=False)

    try:
        trainer.train()
    except Exception as e:
        if str(e) == "EARLY_STOP":
            print(" Early stopping exit cleanly.")
        else:
            raise
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=ShipSARMapper(cfg, is_train=False))
    print("'\n'___Running final validation and visual analysis...'\n'")

    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, cfg.OUTPUT_DIR)  # Deprecated-TODO:change this
    # evaluator = COCOEvaluator(dataset_name=cfg.DATASETS.TEST[0],tasks=("bbox",),distributed=False,
    # output_dir=cfg.OUTPUT_DIR)

    metrics = inference_on_dataset(trainer.model, val_loader, evaluator)

    writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, "tensorboard"))
    writer.add_scalar("Final_Val/AP", metrics["bbox"]["AP"], 0)
    writer.close()

    print("\nFinal Validation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    log_val_predictions_to_tensorboard(cfg, trainer.model, val_loader, os.path.join(output_dir, "tensorboard"))

    prediction_log_path = os.path.join(output_dir, "prediction_log.csv")
    if os.path.exists(prediction_log_path):
        df = pd.read_csv(prediction_log_path)
        class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes
        evaluator = MiniAirEvaluator(df, output_dir, class_names)
        evaluator.plot_canonical_confusion_matrix()
        evaluator.plot_precision_recall_f1_table()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help="Base learning rate")
    parser.add_argument("--batch", type=int, default=512, help="ROI-batch size per image")
    parser.add_argument("--name", type=str, default="default", help="Sweep run name")
    parser.add_argument("--nms", type=float, default=0.5, help="ROI-NMS Threshold")
    parser.add_argument("--score", type=float, default=0.5, help="ROI-Score Threshold")

    args = parser.parse_args()

    output_dir = f"./trained_models_SAR/SAR_sweep_{args.name}"
    json_path = "./json/coco_train.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    num_classes = len(data["categories"])

    setup_and_train(output_dir,
                    num_classes,
                    lr=args.lr,
                    roi_batch=args.batch,
                    nms=args.nms,
                    score=args.score
                    )
