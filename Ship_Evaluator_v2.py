import os
import json
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from contextlib import redirect_stdout
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)


class InfraEvaluator:
    """
    Vehicle evaluator (xView / car-truck setup) built on top of the generic Infra evaluator.
    Expects a `prediction_log.csv` in each inference folder with at least:

        image, gt_class, pred_class, iou, score

    Default classes: ["car", "truck"].
    """

    def __init__(
        self,
        inference_dir: str,
        class_names=None,
        skip_visuals: bool = False,
        skip_plots: bool = False,
        iou_threshold: float = 0.5,
        metrics_filenames=None,
    ):
        self.inference_dir = inference_dir.rstrip("/")
        self.csv_path = os.path.join(self.inference_dir, "prediction_log.csv")
        self.vis_folder = os.path.join(self.inference_dir, "vis")
        self.vis_errors_out = os.path.join(self.inference_dir, "vis_errors")

        # Default: 2-class Vehicles (xView subset). Change here if needed.
        self.class_names = class_names if class_names is not None else ["car", "truck"]

        self.skip_visuals = skip_visuals
        self.skip_plots = skip_plots
        self.iou_threshold = iou_threshold

        # Try both old and new metric filenames
        self.metrics_filenames = (
            metrics_filenames if metrics_filenames is not None else ["metrics.json", "metrics.SIM_json"]
        )

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Prediction log not found at {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------
    def run_all(self):
        log_file = os.path.join(self.inference_dir, "evaluation_log.txt")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        with open(log_file, "w") as log, redirect_stdout(log):
            print("Running evaluation for:", self.inference_dir)

            # Core confusion / detection stats
            self.plot_confusion_stacked_and_grouped()
            self.evaluate_per_class_ap()
            self.plot_grouped_confusion_bars()
            self.plot_canonical_confusion_matrix()
            self.plot_precision_recall_f1_table()

            # Optional visual error extraction
            if not self.skip_visuals:
                self.extract_misclassified_images()

            # Heavier plots
            if not self.skip_plots:
                self.plot_iou_distribution()
                self.plot_iou_mean_bar()
                self.plot_f1_vs_threshold()
                self.plot_max_f1_summary_table()
                self.plot_max_f1_summary_heatmap()
                self.precision_recall_per_class()
                self.plot_precision_recall_curves_per_class()
                self.plot_precision_recall_vs_threshold()
                self.plot_coco_metrics_table()

            # Compact PDF
            self.export_pdf_report()

            print("✅ All evaluation steps completed.")
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_confusion_matrix(self, iou_threshold=None):
        """
        Build (N+1)x(N+1) confusion matrix for a given IoU threshold.
        Last row/col = background.
        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        df = self.df.copy()
        num_classes = len(self.class_names)
        bg_idx = num_classes

        y_true, y_pred = [], []

        for _, row in df.iterrows():
            gt = int(row["gt_class"])
            pred_cls = int(row["pred_class"])
            iou = row["iou"]

            # matched detection (TP or class confusion)
            if gt != -1 and pred_cls != -1 and iou >= iou_threshold:
                y_true.append(gt)
                y_pred.append(pred_cls)

            # FN: GT exists but no valid detection / low IoU / no class
            elif gt != -1 and (pred_cls == -1 or iou < iou_threshold):
                y_true.append(gt)
                y_pred.append(bg_idx)

            # FP: prediction on background (no GT)
            elif gt == -1 and pred_cls != -1:
                y_true.append(bg_idx)
                y_pred.append(pred_cls)

            # gt == -1 and pred == -1 → ignore
            else:
                continue

        labels = list(range(num_classes)) + [bg_idx]
        return confusion_matrix(y_true, y_pred, labels=labels)

    def _assign_match_type(self, df, iou_threshold=None):
        """
        Add a 'match_type' column: TP / FP / FN using the same logic
        as the confusion matrix, but from the GT perspective.
        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        def classify(row):
            gt = row["gt_class"]
            pred = row["pred_class"]
            iou = row["iou"]

            # Any GT that is not correctly detected is an FN
            if gt != -1:
                if pred != -1 and pred == gt and iou >= iou_threshold:
                    return "TP"
                else:
                    return "FN"
            else:
                # No GT: prediction on background → FP
                if pred != -1:
                    return "FP"
                else:
                    return "ignore"

        out = df.copy()
        out["match_type"] = out.apply(classify, axis=1)
        out = out[out["match_type"] != "ignore"]
        return out


    # ------------------------------------------------------------------
    # Per-class pseudo AP summary
    # ------------------------------------------------------------------
    def evaluate_per_class_ap(self):
        df = self.df.copy()
        # keep only actual predictions
        df = df[df["pred_class"] != -1].copy()

        # a detection counts as a "match" only if class + IoU are correct
        df["is_match"] = (
                (df["gt_class"] == df["pred_class"])
                & (df["gt_class"] != -1)
                & (df["iou"] >= self.iou_threshold)
        )

        summary = df.groupby("pred_class").agg(
            total_preds=("image", "count"),
            avg_iou=("iou", "mean"),
            match_rate=("is_match", "mean"),
        ).reset_index()

        summary["class_name"] = summary["pred_class"].map(
            lambda x: self.class_names[x] if 0 <= x < len(self.class_names) else "Unknown"
        )
        summary.to_csv(
            os.path.join(self.inference_dir, "per_class_ap_summary.csv"),
            index=False,
        )

        x = np.arange(len(summary["class_name"]))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, summary["match_rate"], width, label="Match Rate")
        ax.bar(x + width / 2, summary["avg_iou"], width, label="Avg IoU")

        ax.set_ylabel("Score")
        ax.set_title("Match Rate and Average IoU per Class")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["class_name"])
        ax.legend()
        ax.axhline(y=0.85, linestyle="--", linewidth=1)
        ax.text(len(x) - 0.5, 0.855, "ESA Target 85%", fontsize=9)

        fig.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "per_class_ap_summary.png"), dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # Confusion stats: TP/FP/FN per class (stacked + grouped)
    # ------------------------------------------------------------------
    def plot_grouped_confusion_bars(self):
        df = self._assign_match_type(self.df)

        stats = df.groupby(["gt_class", "match_type"]).size().unstack(fill_value=0).reset_index()
        stats["class_name"] = stats["gt_class"].map(
            lambda x: self.class_names[x] if 0 <= x < len(self.class_names) else "Unmatched"
        )

        if "TP" not in stats:
            stats["TP"] = 0
        if "FP" not in stats:
            stats["FP"] = 0
        if "FN" not in stats:
            stats["FN"] = 0

        classes = stats["class_name"].values
        TP = stats["TP"].values
        FP = stats["FP"].values
        FN = stats["FN"].values

        x = np.arange(len(classes))
        width = 0.25

        plt.figure(figsize=(10, 5))
        plt.bar(x - width, TP, width=width, label="TP")
        plt.bar(x, FP, width=width, label="FP")
        plt.bar(x + width, FN, width=width, label="FN")

        plt.xticks(x, classes, rotation=45)
        plt.ylabel("Count")
        plt.title("TP / FP / FN (Grouped View)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "confusion_stats_grouped.png"), dpi=300)
        plt.close()


    def plot_canonical_confusion_matrix(self):
        """
        Canonical confusion matrix (N+1 x N+1) using the row-wise encoding in prediction_log.csv.
        - gt_class: ground truth class index, or -1 for background
        - pred_class: predicted class index, or -1 for "no prediction"
        - iou: IoU between prediction and GT (0 if none)
        """
        df = self.df.copy()
        num_classes = len(self.class_names)
        bg_idx = num_classes
        label_names = self.class_names + ["background"]

        y_true, y_pred = [], []

        for _, row in df.iterrows():
            gt = int(row["gt_class"])
            pred_cls = int(row["pred_class"])
            iou = row["iou"]

            # 1) matched detection (counts as TP or class confusion)
            if gt != -1 and pred_cls != -1 and iou >= self.iou_threshold:
                y_true.append(gt)
                y_pred.append(pred_cls)

            # 2) FN: GT present but no valid detection / low IoU / no predicted class
            elif gt != -1 and (pred_cls == -1 or iou < self.iou_threshold):
                y_true.append(gt)
                y_pred.append(bg_idx)

            # 3) FP: prediction with no GT (gt == -1)
            elif gt == -1 and pred_cls != -1:
                y_true.append(bg_idx)
                y_pred.append(pred_cls)

            # (gt == -1 and pred == -1) – meaningless row, ignore
            else:
                continue

        labels = list(range(num_classes)) + [bg_idx]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(xticks_rotation=45, cmap="Purples", values_format="d")
        plt.title(f"Full Confusion Matrix\nIoU ≥ {self.iou_threshold:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "confusion_matrix_canonical.png"), dpi=300)
        plt.close()

        df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
        df_cm.to_csv(os.path.join(self.inference_dir, "confusion_matrix_canonical.csv"))

        core_cm = cm[:num_classes, :num_classes]
        accuracy = np.trace(core_cm) / np.sum(core_cm) if np.sum(core_cm) > 0 else 0.0
        with open(os.path.join(self.inference_dir, "accuracy.txt"), "w") as f:
            f.write("Canonical Confusion Matrix with Background and Class Confusions\n")
            f.write(f"IoU ≥ {self.iou_threshold:.2f}\n")
            f.write(f"Overall Accuracy (no BG): {accuracy:.4f}\n")

    # ------------------------------------------------------------------
    # Canonical object detection confusion matrix (N+1 x N+1)
    # ------------------------------------------------------------------
    def plot_precision_recall_f1_table(self):
        cm_path = os.path.join(self.inference_dir, "confusion_matrix_canonical.csv")
        if not os.path.exists(cm_path):
            print("❌ Canonical confusion matrix not found.")
            return

        df_cm = pd.read_csv(cm_path, index_col=0)
        num_classes = len(self.class_names)

        results = []
        for i in range(num_classes):
            TP = df_cm.iloc[i, i]
            row_sum = df_cm.iloc[i, :].sum()
            col_sum = df_cm.iloc[:, i].sum()

            FP = col_sum - TP  # all predicted as class i but GT != i
            FN = row_sum - TP  # all GT i but predicted != i

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            acc = TP / row_sum if row_sum > 0 else 0.0  # per-class accuracy

            results.append(
                {
                    "Class": self.class_names[i],
                    "TP": int(TP),
                    "FP": int(FP),
                    "FN": int(FN),
                    "Precision": round(precision, 3),
                    "Recall": round(recall, 3),
                    "F1 Score": round(f1, 3),
                    "Accuracy": round(acc, 3),
                }
            )

        df_results = pd.DataFrame(results)

        # Micro / macro summaries (micro = global TP/FP/FN)
        total_TP = df_results["TP"].sum()
        total_FP = df_results["FP"].sum()
        total_FN = df_results["FN"].sum()
        total_GT = total_TP + total_FN

        micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )
        micro_accuracy = total_TP / total_GT if total_GT > 0 else 0.0

        summary_row = {
            "Class": "Model Total",
            "TP": int(total_TP),
            "FP": int(total_FP),
            "FN": int(total_FN),
            "Precision": round(micro_precision, 3),
            "Recall": round(micro_recall, 3),
            "F1 Score": round(micro_f1, 3),
            "Accuracy": round(micro_accuracy, 3),
        }
        df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)

        plt.figure(figsize=(10, len(df_results) * 0.6))
        sns.heatmap(
            df_results.set_index("Class"),
            annot=True,
            fmt=".3f",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
        )
        plt.title("Precision / Recall / F1 / Accuracy per Class", fontsize=14)
        plt.tight_layout()
        plt.figtext(
            0.5,
            -0.03,
            "Accuracy = TP / (TP + FN) per class.\n"
            "FP and FN include both background errors and inter-class confusions.",
            wrap=True,
            horizontalalignment="center",
            fontsize=9,
            color="gray",
        )
        plt.savefig(os.path.join(self.inference_dir, "precision_recall_f1_table.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Precision / Recall / F1 table from canonical CM
    # ------------------------------------------------------------------
    def plot_confusion_stacked_and_grouped(self):
        df = self._assign_match_type(self.df)

        stats = df.groupby(["gt_class", "match_type"]).size().unstack(fill_value=0).reset_index()
        stats["class_name"] = stats["gt_class"].map(
            lambda x: self.class_names[x] if 0 <= x < len(self.class_names) else "FP (Unmatched)"
        )

        # ensure we always have columns
        if "TP" not in stats:
            stats["TP"] = 0
        if "FP" not in stats:
            stats["FP"] = 0
        if "FN" not in stats:
            stats["FN"] = 0

        total = stats["TP"] + stats["FP"] + stats["FN"]
        stats["FN_rate"] = stats["FN"] / total.replace({0: np.nan})
        stats = stats.sort_values("FN_rate", ascending=False)

        classes = stats["class_name"].values
        TP = stats["TP"].values
        FP = stats["FP"].values
        FN = stats["FN"].values

        # ----- stacked -----
        plt.figure(figsize=(10, 5))
        plt.bar(classes, TP, label="TP")
        plt.bar(classes, FP, bottom=TP, label="FP")
        plt.bar(classes, FN, bottom=TP + FP, label="FN")

        for i, (tp, fp, fn) in enumerate(zip(TP, FP, FN)):
            total_i = tp + fp + fn
            if total_i == 0:
                continue
            y_offset = 0
            for val, label in zip([tp, fp, fn], ["TP", "FP", "FN"]):
                if val > 0:
                    percent = int(round((val / total_i) * 100))
                    plt.text(
                        i,
                        y_offset + val / 2,
                        f"{label} {val} ({percent}%)",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                    )
                    y_offset += val

        plt.ylabel("Count")
        plt.title("Per-Class Detection Outcome Breakdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "confusion_stats_stacked.png"), dpi=300)
        plt.close()

        # ----- grouped -----
        x = np.arange(len(classes))
        width = 0.25

        plt.figure(figsize=(10, 5))
        plt.bar(x - width, TP, width=width, label="TP")
        plt.bar(x, FP, width=width, label="FP")
        plt.bar(x + width, FN, width=width, label="FN")

        plt.xticks(x, classes, rotation=45)
        plt.ylabel("Count")
        plt.title("TP / FP / FN (Grouped View)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "confusion_stats_grouped.png"), dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # Visual error extraction
    # ------------------------------------------------------------------
    def extract_misclassified_images(self):
        df = self._assign_match_type(self.df)

        os.makedirs(self.vis_errors_out, exist_ok=True)
        misclassified = df[df["match_type"] != "TP"]
        for filename in misclassified["image"].unique():
            src = os.path.join(self.vis_folder, filename)
            dst = os.path.join(self.vis_errors_out, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)

    # ------------------------------------------------------------------
    # IoU plots
    # ------------------------------------------------------------------
    def plot_iou_distribution(self):
        df = self.df[self.df["iou"] > 0].copy()
        df = self._assign_match_type(df)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="pred_class", y="iou", hue="match_type")
        plt.title("IoU Distribution by Predicted Class and Match Type")
        plt.xlabel("Predicted Class")
        plt.ylabel("IoU")
        plt.xticks(ticks=range(len(self.class_names)), labels=self.class_names)
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "iou_distribution.png"), dpi=300)
        plt.close()

    def plot_iou_mean_bar(self):
        df = self.df[self.df["iou"] > 0]
        mean_ious = df[df["gt_class"] != -1].groupby("gt_class")["iou"].mean()

        valid_indexes = [i for i in mean_ious.index if 0 <= i < len(self.class_names)]
        valid_class_names = [self.class_names[i] for i in valid_indexes]
        valid_ious = [mean_ious[i] for i in valid_indexes]

        plt.figure(figsize=(8, 5))
        plt.bar(valid_class_names, valid_ious)
        plt.title("Mean IoU per Class")
        plt.xlabel("Class")
        plt.ylabel("Mean IoU")
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "mean_iou_per_class.png"), dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # F1 vs IoU threshold
    # ------------------------------------------------------------------
    def plot_f1_vs_threshold(self):
        thresholds = np.linspace(0.1, 0.9, 9)
        f1_results = []

        num_classes = len(self.class_names)

        for t in thresholds:
            cm = self._compute_confusion_matrix(iou_threshold=t)

            for cls_id in range(num_classes):
                TP = cm[cls_id, cls_id]
                row_sum = cm[cls_id, :].sum()
                col_sum = cm[:, cls_id].sum()

                FP = col_sum - TP
                FN = row_sum - TP

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                f1_results.append(
                    {
                        "class_id": cls_id,
                        "class_name": self.class_names[cls_id],
                        "threshold": t,
                        "F1": f1,
                    }
                )

        df_f1 = pd.DataFrame(f1_results)
        plt.figure(figsize=(10, 6))
        for cls_id in range(num_classes):
            subset = df_f1[df_f1["class_id"] == cls_id]
            plt.plot(subset["threshold"], subset["F1"], label=self.class_names[cls_id])

        plt.title("F1 Score per Class over IoU Thresholds")
        plt.xlabel("IoU Threshold")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "f1_score_per_class.png"), dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # Max-F1 summary over IoU thresholds
    # ------------------------------------------------------------------
    def plot_max_f1_summary_table(self):
        thresholds = [round(x, 2) for x in np.linspace(0.1, 0.9, 9)]
        num_classes = len(self.class_names)

        best = {
            i: {"max_F1": 0.0, "threshold": None, "TP": 0, "FP": 0, "FN": 0}
            for i in range(num_classes)
        }

        for t in thresholds:
            cm = self._compute_confusion_matrix(iou_threshold=t)

            for i in range(num_classes):
                TP = cm[i, i]
                row_sum = cm[i, :].sum()
                col_sum = cm[:, i].sum()

                FP = col_sum - TP
                FN = row_sum - TP

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                if f1 > best[i]["max_F1"]:
                    best[i] = {
                        "max_F1": f1,
                        "threshold": t,
                        "TP": int(TP),
                        "FP": int(FP),
                        "FN": int(FN),
                    }

        best_results = []
        for i in range(num_classes):
            b = best[i]
            best_results.append(
                {
                    "class_id": i,
                    "class_name": self.class_names[i],
                    "max_F1": round(b["max_F1"], 4),
                    "threshold": b["threshold"],
                    "TP": b["TP"],
                    "FP": b["FP"],
                    "FN": b["FN"],
                }
            )

        df_summary = pd.DataFrame(best_results)
        df_summary.to_csv(os.path.join(self.inference_dir, "max_f1_summary_table.csv"), index=False)
    # ------------------------------------------------------------------
    # plot_max_f1_summary_heatmap
    # ------------------------------------------------------------------
    def plot_max_f1_summary_heatmap(self):
        path = os.path.join(self.inference_dir, "max_f1_summary_table.csv")
        if not os.path.exists(path):
            print(" Cannot find max_f1_summary_table.csv. Please run plot_max_f1_summary_table() first.")
            return

        df = pd.read_csv(path).set_index("class_name")[["max_F1", "threshold", "TP", "FP", "FN"]]
        plt.figure(figsize=(10, 3.5))
        sns.heatmap(
            df,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar=False,
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title("Max F1 Summary (per Class)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "max_f1_summary_table.png"), dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # PR per class (bar + curve)
    # ------------------------------------------------------------------
    def precision_recall_per_class(self):
        cm_path = os.path.join(self.inference_dir, "confusion_matrix_canonical.csv")
        if not os.path.exists(cm_path):
            print("❌ Canonical confusion matrix not found. Please run plot_canonical_confusion_matrix() first.")
            return

        df_cm = pd.read_csv(cm_path, index_col=0)
        num_classes = len(self.class_names)

        stats = []
        for i in range(num_classes):
            TP = df_cm.iloc[i, i]
            row_sum = df_cm.iloc[i, :].sum()
            col_sum = df_cm.iloc[:, i].sum()

            FP = col_sum - TP
            FN = row_sum - TP

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            stats.append(
                {
                    "class_id": i,
                    "class_name": self.class_names[i],
                    "TP": int(TP),
                    "FP": int(FP),
                    "FN": int(FN),
                    "Precision": round(precision, 3),
                    "Recall": round(recall, 3),
                    "F1": round(f1, 3),
                }
            )

        df_stats = pd.DataFrame(stats)
        df_stats.to_csv(os.path.join(self.inference_dir, "precision_recall_per_class.csv"), index=False)

        x = np.arange(len(self.class_names))
        width = 0.20
        fig, ax = plt.subplots(figsize=(10, 5))

        precisions = df_stats["Precision"].values
        recalls = df_stats["Recall"].values
        f1s = df_stats["F1"].values

        ax.bar(x - width, precisions, width, label="Precision")
        ax.bar(x, recalls, width, label="Recall")
        ax.bar(x + width, f1s, width, label="F1 Score")

        for i in range(len(self.class_names)):
            ax.text(x[i] - width, precisions[i] + 0.01, f"{precisions[i]:.3f}", ha="center", fontsize=8)
            ax.text(x[i], recalls[i] + 0.01, f"{recalls[i]:.3f}", ha="center", fontsize=8)
            ax.text(x[i] + width, f1s[i] + 0.01, f"{f1s[i]:.3f}", ha="center", fontsize=8)

        ax.set_ylabel("Score")
        ax.set_title("Precision, Recall, and F1 Score per Class")
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.axhline(y=0.85, color="gray", linestyle="--", linewidth=1)
        ax.text(len(x) - 0.5, 0.857, "ESA Target 85%", color="gray", fontsize=9)

        fig.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "precision_recall_per_class.png"), dpi=300)
        plt.close()

    def plot_precision_recall_curves_per_class(self):
        df = self.df.copy()
        if "score" not in df.columns:
            print(" No 'score' column in prediction_log.csv, skipping PR curves.")
            return

        plt.figure(figsize=(10, 6))

        for cls_id, cls_name in enumerate(self.class_names):
            df_cls = df[df["pred_class"] == cls_id].copy()
            if df_cls.empty:
                continue

            # positive if this prediction is a correct detection for class cls_id
            y_true = (
                    (df_cls["gt_class"] == cls_id)
                    & (df_cls["iou"] >= self.iou_threshold)
            ).astype(int)
            y_score = df_cls["score"]

            if y_true.sum() == 0:
                continue

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)

            plt.plot(recall, precision, label=f"{cls_name} (AP={ap:.2f})")
            plt.fill_between(recall, precision, alpha=0.1)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves per Class")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "precision_recall_per_class_curves.png"), dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # PR / F1 vs confidence threshold
    # ------------------------------------------------------------------
    def plot_precision_recall_vs_threshold(self):
        df = self.df.copy()
        if "score" not in df.columns:
            print("❌ No 'score' column in prediction_log.csv, skipping PR vs threshold.")
            return

        thresholds = np.linspace(0.0, 1.0, 101)
        precisions, recalls, f1s = [], [], []

        total_gt = len(df[df["gt_class"] != -1])

        for t in thresholds:
            df_thresh = df[df["score"] >= t].copy()

            # TP: correct class + IoU above threshold
            is_tp = (
                    (df_thresh["gt_class"] != -1)
                    & (df_thresh["pred_class"] != -1)
                    & (df_thresh["gt_class"] == df_thresh["pred_class"])
                    & (df_thresh["iou"] >= self.iou_threshold)
            )
            TP = int(is_tp.sum())

            # Any prediction that is not a TP is an FP (including class confusions)
            num_preds = int((df_thresh["pred_class"] != -1).sum())
            FP = num_preds - TP

            FN = total_gt - TP

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, precisions, label="Precision")
        plt.plot(thresholds, recalls, label="Recall")
        plt.plot(thresholds, f1s, label="F1 Score")

        best_f1 = max(f1s)
        best_idx = f1s.index(best_f1)
        best_threshold = thresholds[best_idx]
        plt.axvline(x=best_threshold, linestyle="--", linewidth=1)
        plt.text(best_threshold + 0.01, best_f1, f"Best F1 = {best_f1:.3f}", fontsize=9)

        plt.title("Precision / Recall / F1 vs Confidence Threshold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "pr_f1_vs_threshold.png"), dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # COCO metrics table (AP, AP50, AP75, etc.)
    # ------------------------------------------------------------------
    def _find_metrics_file(self):
        for name in self.metrics_filenames:
            path = os.path.join(self.inference_dir, name)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path
        return None

    def plot_coco_metrics_table(self):
        import seaborn as sns
        import matplotlib.pyplot as plt

        path = self._find_metrics_file()
        if path is None:
            print(f" No non-empty metrics file found (tried: {self.metrics_filenames})")
            return

        with open(path, "r") as f:
            content = f.read().strip()
            if not content:
                print(f" metrics file has no content: {path}")
                return
            try:
                raw_data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON in {path}: {e}")
                return

        bbox_data = raw_data.get("bbox", {})
        if not bbox_data:
            print(f" No bbox section found in metrics file at {path}")
            return

        table_rows = []
        for k in ["AP", "AP50", "AP75", "APs", "APm", "APl"]:
            if k in bbox_data:
                table_rows.append({"Metric": k, "Value": round(bbox_data[k], 3)})

        for k, v in bbox_data.items():
            if k.startswith("AP-"):
                table_rows.append({"Metric": k, "Value": round(v, 3)})

        df = pd.DataFrame(table_rows)
        if df.empty:
            print(f" Empty metrics table for {path}")
            return

        plt.figure(figsize=(7, len(df) * 0.4))
        sns.heatmap(df.set_index("Metric"), annot=True, fmt=".3f", cmap="YlGnBu", cbar=False, linewidths=0.5)
        plt.title("COCO Evaluation Metrics (bbox)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "coco_metrics_table.png"), dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # PDF report assembling key plots
    # ------------------------------------------------------------------
    def export_pdf_report(self):
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg

        model_name = os.path.basename(self.inference_dir.rstrip("/"))
        pdf_path = os.path.join(self.inference_dir, f"evaluation_summary_{model_name}.pdf")

        plots = [
            "confusion_matrix_canonical.png",
            "confusion_stats_stacked.png",
            "precision_recall_per_class.png",
            "precision_recall_f1_table.png",
            "coco_metrics_table.png",
        ]

        with PdfPages(pdf_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            plt.axis("off")
            plt.text(0.5, 0.6, "Evaluation Report", ha="center", va="center", fontsize=24, weight="bold")
            plt.text(0.5, 0.45, f"Model: {model_name}", ha="center", va="center", fontsize=16)
            plt.text(0.5, 0.3, f"Path: {self.inference_dir}", ha="center", va="center", fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

            for plot in plots:
                path = os.path.join(self.inference_dir, plot)
                if os.path.exists(path):
                    fig = plt.figure(figsize=(12, 7))
                    img = mpimg.imread(path)
                    plt.imshow(img, interpolation="none")
                    plt.axis("off")
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

        print(f"✅ Evaluation summary PDF saved to: {pdf_path}")


# ----------------------------------------------------------------------
# Batch runner
# ----------------------------------------------------------------------
def batch_run_all(base_dir, skip_visuals=False, skip_plots=False, class_names=None):
    print(f" Running batch evaluation in: {base_dir}\n")
    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue

        csv_path = os.path.join(full_path, "prediction_log.csv")
        # csv_path = os.path.join(full_path, "vis2predictions.csv")
        if not os.path.exists(csv_path):
            continue

        print(f" Evaluating folder: {folder}")
        try:
            evaluator = InfraEvaluator(
                full_path,
                class_names=class_names,
                skip_visuals=skip_visuals,
                skip_plots=skip_plots,
            )
            evaluator.run_all()
        except Exception as e:
            print(f" Failed in {folder}: {e}")
    print("\n Batch evaluation completed.")


if __name__ == "__main__":
    # EDIT THIS to your base directory of inference results
    # base_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipSAR/run_0/finetune_december_800x800/SAR_sweep_v5_lr0.0002_b512_/output_"
    # base_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/fine_tune_lr15_b512_/Optical_sweep_default/output/500/"
    # batch_run_all(base_dir, skip_visuals=True, skip_plots=False, class_names=["ship"])
    #model bench
    # base_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/Optical_sweep_lr0.00015_b512_s0.5_nms0.4/ben_model_benchmark/"
    # base_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/fine_tune_lr15_b512_/Optical_sweep_default/fine_tune768/"
    base_dir = "/media/gpaps/My Passport/CVRL-GeorgeP/_/annotations_v2_10_2025/optical/planes/images/output_pred_bench_4/"
    batch_run_all(base_dir, skip_visuals=True, skip_plots=False, class_names=['aircraft','helicopter'])#class_names=["Commercial", "Military", "Submarines", "Recreational Boats", "Fishing Boats"]
# )
