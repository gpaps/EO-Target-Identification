import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from detectron2.engine import HookBase
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader


class MiniAirEvaluator:
    """
    Calculates Precision, Recall, and Generates Confusion Matrix
    """

    def __init__(self, df, output_dir, class_names):
        self.df = df.copy()
        self.output_dir = output_dir
        self.class_names = class_names
        self.bg_idx = len(class_names)

    def compute_metrics(self):
        gt_df = self.df[self.df['gt_class'] != -1]
        results = {}
        for cls_id, cls_name in enumerate(self.class_names):
            cls_data = gt_df[gt_df['gt_class'] == cls_id]
            total = len(cls_data)
            detected = len(cls_data[cls_data['pred_class'] != -1])
            recall = detected / total if total > 0 else 0
            results[cls_name] = {'Recall': recall, 'Total': total}
        return pd.DataFrame(results).T

    def plot_confusion_matrix(self):
        y_true = self.df['gt_class'].values
        y_pred = self.df['pred_class'].values
        y_pred[y_pred == -1] = self.bg_idx
        y_true[y_true == -1] = self.bg_idx

        labels = self.class_names + ["Background"]
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
        plt.title('Confusion Matrix (Validation)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()


def dump_predictions_to_csv(model, data_loader, output_path):
    print(f"[Info] Dumping predictions to {output_path}...")
    model.eval()
    results = []

    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs)
            for input_im, output_im in zip(inputs, outputs):
                img_name = input_im["file_name"]
                gt_instances = input_im["instances"]
                gt_boxes = gt_instances.gt_boxes.tensor.cpu().numpy()
                gt_classes = gt_instances.gt_classes.cpu().numpy()

                pred_instances = output_im["instances"]
                pred_boxes = pred_instances.pred_boxes.tensor.cpu().numpy()
                pred_classes = pred_instances.pred_classes.cpu().numpy()
                scores = pred_instances.scores.cpu().numpy()

                if len(gt_classes) > 0:
                    for i, gt_cls in enumerate(gt_classes):
                        best_iou = 0
                        best_match_cls = -1
                        best_score = 0
                        gt_box = gt_boxes[i]

                        for j, pred_box in enumerate(pred_boxes):
                            xA = max(gt_box[0], pred_box[0])
                            yA = max(gt_box[1], pred_box[1])
                            xB = min(gt_box[2], pred_box[2])
                            yB = min(gt_box[3], pred_box[3])
                            interArea = max(0, xB - xA) * max(0, yB - yA)
                            boxAArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                            boxBArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                            iou = interArea / float(boxAArea + boxBArea - interArea)

                            if iou > 0.5 and iou > best_iou:
                                best_iou = iou
                                best_match_cls = pred_classes[j]
                                best_score = scores[j]

                        results.append({
                            "image": os.path.basename(img_name),
                            "gt_class": int(gt_cls),
                            "pred_class": int(best_match_cls),
                            "score": float(best_score),
                            "iou": float(best_iou)
                        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


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
            results = inference_on_dataset(
                self.model,
                self.val_loader,
                COCOEvaluator(self.trainer.cfg.DATASETS.TEST[0], self.trainer.cfg, False, self.output_dir)
            )
            print(f"[EvalHook] Eval results at iter {next_iter}: {results}")
            current_ap = results.get("bbox", {}).get("AP", -1)

            if current_ap > self.best_ap:
                self.best_ap = current_ap
                self.counter = 0
                checkpointer = DetectionCheckpointer(self.model, self.output_dir)
                checkpointer.save("model_best")
            else:
                self.counter += 1
                print(f"[EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print(" Early stopping triggered.")
                self.trainer.storage.put_scalars(early_stop_iter=next_iter)
                raise Exception("EARLY_STOP")


def plot_loss_curves(output_dir):
    """
    Reads metrics.json and saves a grid plot of all losses to 'training_loss_summary.png'.
    """
    json_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(json_path):
        print(f"[Warn] No metrics.json found at {json_path}")
        return

    # Read the JSON line by line
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass

    df = pd.DataFrame(data)

    # Filter for rows that actually contain training loss
    if 'total_loss' not in df.columns:
        print("[Warn] metrics.json does not contain 'total_loss'.")
        return

    df_train = df[df['total_loss'].notna()]

    # Define the losses we want to visualize
    metrics = ['total_loss', 'loss_cls', 'loss_box_reg', 'loss_rpn_cls', 'loss_rpn_loc']

    # Create a nice 2x3 Grid
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"Training Loss Summary: {os.path.basename(output_dir)}", fontsize=16)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if metric in df_train.columns:
            # Plot the raw data with transparency
            sns.lineplot(data=df_train, x='iteration', y=metric, ax=axes[i], alpha=0.3, color='blue',
                         label='Raw')
            # Plot a smoothed trend line (Moving Average)
            df_train[f'{metric}_smooth'] = df_train[metric].rolling(window=50).mean()
            sns.lineplot(data=df_train, x='iteration', y=f'{metric}_smooth', ax=axes[i], linewidth=2,
                         color='red', label='Smoothed (MA50)')

            axes[i].set_title(metric.replace('_', ' ').upper(), fontsize=12)
            axes[i].set_xlabel("Iteration")
            axes[i].set_ylabel("Loss")
            axes[i].legend()

    # Delete unused subplot (since we have 5 plots in a grid of 6)
    if len(metrics) < 6:
        fig.delaxes(axes[5])

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_loss_summary.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Loss Summary Plot saved to {save_path}")