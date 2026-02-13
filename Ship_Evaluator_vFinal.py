import os, json, sys
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from sklearn.metrics import precision_recall_curve
from mpmath.libmp import normalize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from triton.language.extra.hip.libdevice import trunc


class InfraEvaluator:
    def __init__(self, inference_dir: str, class_names=None, skip_visuals=False, skip_plots=False):
        self.inference_dir = inference_dir.rstrip("/")
        self.csv_path = os.path.join(self.inference_dir, "prediction_log.csv")
        self.vis_folder = os.path.join(self.inference_dir, "vis")
        self.vis_errors_out = os.path.join(self.inference_dir, "vis_errors")
        self.class_names = class_names if class_names else ["Commercial", "Military", "Submarines", "Recreational Boats", "Fishing Boats"]  # <---- Optical---->
        #["ship"]#["tank", "truck"]
        self.skip_visuals = skip_visuals
        self.skip_plots = skip_plots

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Prediction log not found at {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

    def run_all(self):
        log_file = os.path.join(self.inference_dir, "evaluation_log.txt")
        with open(log_file, "w") as log, redirect_stdout(log):
            print(" Running evaluation for:", self.inference_dir)
            # self.evaluate_confusion_stats()
            # self.plot_confusion_stacked()
            # self.plot_confusion_matrix() confussion full
            self.plot_confusion_stacked_and_grouped()
            self.evaluate_per_class_ap()
            # self.plot_grouped_confusion_bars()
            self.plot_canonical_confusion_matrix()
            self.plot_precision_recall_f1_table()
            # self.extract_misclassified_images()
            self.plot_precision_recall_vs_threshold()
            self.plot_binary_class_metrics()
            self.plot_iou_distribution()
            self.plot_iou_mean_bar()
            self.plot_f1_vs_threshold()
            self.plot_max_f1_summary_table()
            self.plot_max_f1_summary_heatmap()
            self.precision_recall_per_class()
            self.plot_precision_recall_curves_per_class()
            self.plot_coco_metrics_table()
            self.export_pdf_report()
            print("✅ All evaluation steps completed.")

    def evaluate_per_class_ap(self):
        df = self.df.copy()
        df = df[df["pred_class"] != -1]
        df["is_match"] = df["gt_class"] == df["pred_class"]

        summary = df.groupby("pred_class").agg(
            total_preds=("image", "count"),
            avg_iou=("iou", "mean"),
            match_rate=("is_match", "mean")
        ).reset_index()

        summary["class_name"] = summary["pred_class"].map(
            lambda x: self.class_names[x] if 0 <= x < len(self.class_names) else "Unknown"
        )
        summary.to_csv(os.path.join(self.inference_dir, "per_class_ap_summary.csv"), index=False)

        x = np.arange(len(summary["class_name"]))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - width / 2, summary["match_rate"], width, label="Match Rate", color="skyblue")
        bars2 = ax.bar(x + width / 2, summary["avg_iou"], width, label="Avg IoU", color="salmon")

        ax.set_ylabel("Score")
        ax.set_title("Match Rate and Average IoU per Class")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["class_name"])
        ax.legend()
        ax.axhline(y=0.85, color="gray", linestyle="--", linewidth=1)
        ax.text(len(x) - 0.5, 0.855, "ESA Target 85%", color="gray", fontsize=9)

        fig.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "per_class_ap_summary.png"), dpi=300)
        plt.close()

    #TODO delete if not nedded
    # def evaluate_confusion_stats(self):
    #     stats = self.df.groupby(["gt_class", "match_type"]).size().unstack(fill_value=0).reset_index()
    ## stats["class_name"] = stats["gt_class"].map(lambda x: self.class_names[x] if x != -1 else "Unmatched")
    # stats["class_name"] = stats["gt_class"].map(
    #     lambda x: self.class_names[x] if 0 <= x < len(self.class_names) else "Unmatched")
    #
    # stats.to_csv(os.path.join(self.inference_dir, "confusion_stats.csv"), index=False)

    # def evaluate_confusion_stats(self):
    #     df = self.df.copy()
    #     df["match_type"] = df.apply(lambda row: (
    #         "TP" if row["gt_class"] == row["pred_class"] and row["gt_class"] != -1 else
    #         "FP" if row["gt_class"] == -1 else
    #         "FN"
    #     ), axis=1)
    #
    #     conf_stats = df.groupby(["gt_class", "match_type"]).size().unstack(fill_value=0).reset_index()
    #     conf_stats["class_name"] = conf_stats["gt_class"].map(
    #         lambda x: self.class_names[x] if (x != -1 and x < len(self.class_names)) else "FP (Unmatched)")
    #     conf_stats.to_csv(os.path.join(self.inference_dir, "confusion_stats.csv"), index=False)
    #
    #     # Visualization
    #     classes = conf_stats["class_name"]
    #     TP = conf_stats.get("TP", [0] * len(classes))
    #     FP = conf_stats.get("FP", [0] * len(classes))
    #     FN = conf_stats.get("FN", [0] * len(classes))
    #
    #     plt.figure(figsize=(10, 5))
    #     plt.bar(classes, TP, label="TP", color="green")
    #     plt.bar(classes, FP, bottom=TP, label="FP", color="red")
    #     plt.bar(classes, FN, bottom=TP + FP, label="FN", color="orange")
    #     plt.ylabel("Count")
    #     plt.title("TP / FP / FN per Class")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.inference_dir, "confusion_stats_stacked999.png"))
    #     plt.close()
    '''Also Delete if not needed'''

    # def plot_confusion_matrix(self):
    #     df = self.df.copy()
    #     df["match_type"] = df.apply(lambda row: (
    #         "TP" if row["gt_class"] == row["pred_class"] and row["gt_class"] != -1 else
    #         "FP" if row["gt_class"] == -1 else "FN"
    #     ), axis=1)
    #
    #     unmatched_label = len(self.class_names)
    #     df["gt_mapped"] = df["gt_class"].apply(lambda x: x if x != -1 else unmatched_label)
    #     df["pred_mapped"] = df["pred_class"]
    #
    #     labels = list(range(len(self.class_names))) + [unmatched_label]
    #     label_names = self.class_names + ["Unmatched"]
    #
    #     # --- Full Confusion Matrix (incl. FP/FN) ---
    #     cm = confusion_matrix(df["gt_mapped"], df["pred_mapped"], labels=labels)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    #     disp.plot(xticks_rotation=45, cmap="Oranges")
    #     plt.title("Full Confusion Matrix (Incl. FP/FN)")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.inference_dir, "confusion_matrix_full.png"))
    #     plt.close()
    #
    #     # --- Matched-Only Confusion Matrix (GT known) ---
    #     y_true_clean = df[df["gt_class"] != -1]["gt_class"]
    #     y_pred_clean = df[df["gt_class"] != -1]["pred_class"]
    #     cm_clean = confusion_matrix(y_true_clean, y_pred_clean, labels=list(range(len(self.class_names))))
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm_clean, display_labels=self.class_names)
    #     disp.plot(xticks_rotation=45, cmap="Blues")
    #     plt.title("Confusion Matrix (IoU ≥ 0.5, Matched Only)")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.inference_dir, "confusion_matrix.png"))
    #     plt.close()

    # def plot_grouped_confusion_bars(self):
    #     df = self.df.copy()
    #     df["match_type"] = df.apply(lambda row: (
    #         "TP" if row["gt_class"] == row["pred_class"] and row["gt_class"] != -1 else
    #         "FP" if row["gt_class"] == -1 else "FN"
    #     ), axis=1)
    #
    #     stats = df.groupby(["gt_class", "match_type"]).size().unstack(fill_value=0).reset_index()
    #     stats["class_name"] = stats["gt_class"].map(
    #         lambda x: self.class_names[x] if 0 <= x < len(self.class_names) else "Unmatched"
    #     )
    #
    #     classes = stats["class_name"]
    #     TP = stats.get("TP", [0] * len(classes))
    #     FP = stats.get("FP", [0] * len(classes))
    #     FN = stats.get("FN", [0] * len(classes))
    #
    #     x = np.arange(len(classes))
    #     width = 0.25
    #
    #     plt.figure(figsize=(10, 5))
    #     plt.bar(x - width, TP, width=width, label="TP", color="green")
    #     plt.bar(x, FP, width=width, label="FP", color="red")
    #     plt.bar(x + width, FN, width=width, label="FN", color="orange")
    #
    #     plt.xticks(x, classes, rotation=45)
    #     plt.ylabel("Count")
    #     plt.title("TP / FP / FN (Grouped View)")
    #     plt.legend()
    #     plt.tight_layout()
    #     # plt.savefig(os.path.join(self.inference_dir, "confusion_stats_grouped.png"), dpi=300)
    #     plt.close()

    def plot_confusion_stacked_and_grouped(self):
        df = self.df.copy()
        df["match_type"] = df.apply(lambda row: (
            "TP" if row["gt_class"] == row["pred_class"] and row["gt_class"] != -1 else
            "FP" if row["gt_class"] == -1 else "FN"
        ), axis=1)

        stats = df.groupby(["gt_class", "match_type"]).size().unstack(fill_value=0).reset_index()
        stats["class_name"] = stats["gt_class"].map(
            lambda x: self.class_names[x] if 0 <= x < len(self.class_names) else "FP (Unmatched)"
        )

        # stats = stats.sort_values("class_name" if "FP (Unmatched)" not in self.class_names else "gt_class")
        stats['FN_rate'] = stats['FN'] / (stats['TP'] + stats['FN'] + stats['FP'])
        stats = stats.sort_values('FN_rate', ascending=False)

        classes = stats["class_name"]
        TP = stats.get("TP", [0] * len(classes))
        FP = stats.get("FP", [0] * len(classes))
        FN = stats.get("FN", [0] * len(classes))

        # -------- Plot 1: Stacked Bar with % labels --------
        plt.figure(figsize=(10, 5))
        bar1 = plt.bar(classes, TP, label="TP", color="green")
        bar2 = plt.bar(classes, FP, bottom=TP, label="FP", color="red")
        bar3 = plt.bar(classes, FN, bottom=TP + FP, label="FN", color="orange")

        for i, (tp, fp, fn) in enumerate(zip(TP, FP, FN)):
            total = tp + fp + fn
            if total == 0:
                continue
            y_offset = 0
            for val, color, label in zip([tp, fp, fn], ["green", "red", "orange"], ["TP", "FP", "FN"]):
                if val > 0:
                    percent = int(round((val / total) * 100))
                    plt.text(i, y_offset + val / 2, f"{label} {val} ({percent}%)", ha="center", va="center", fontsize=9,
                             color="white")

                    # plt.text(i, y_offset + val / 2, f"{label} ({percent}%)", ha="center", va="center", fontsize=8,
                    #          color="white")
                    y_offset += val

        plt.ylabel("Count")
        plt.title("Per-Class Detection Outcome Breakdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "confusion_stats_stacked.png"))
        plt.close()

        # -------- Plot 2: Grouped Bar Chart --------
        x = np.arange(len(classes))
        width = 0.25

        plt.figure(figsize=(10, 5))
        plt.bar(x - width, TP, width=width, label="TP", color="green")
        plt.bar(x, FP, width=width, label="FP", color="red")
        plt.bar(x + width, FN, width=width, label="FN", color="orange")

        plt.xticks(x, classes, rotation=45)
        plt.ylabel("Count")
        plt.title("TP / FP / FN (Grouped View)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "confusion_stats_grouped.png"))
        plt.close()

    # def plot_confusion_stacked(self):
    #     df = self.df.copy()
    #     df["match_type"] = df.apply(lambda row: (
    #         "TP" if row["gt_class"] == row["pred_class"] and row["gt_class"] != -1 else
    #         "FP" if row["gt_class"] == -1 else "FN"
    #     ), axis=1)
    #
    #     confusion_df = df.groupby(["gt_class", "match_type"]).size().unstack(fill_value=0).reset_index()
    #     confusion_df["class_name"] = confusion_df["gt_class"].map(
    #         lambda x: self.class_names[x] if 0 <= x < len(self.class_names) else "FP (Unmatched)"
    #     )
    #
    #     # Sort to move 'FP (Unmatched)' last
    #     confusion_df = confusion_df.sort_values(
    #         "class_name" if "FP (Unmatched)" not in self.class_names else "gt_class")
    #
    #     classes = confusion_df["class_name"]
    #     TP = confusion_df.get("TP", [0] * len(classes))
    #     FP = confusion_df.get("FP", [0] * len(classes))
    #     FN = confusion_df.get("FN", [0] * len(classes))
    #
    #     plt.figure(figsize=(10, 5))
    #     bar1 = plt.bar(classes, TP, label="TP", color="green")
    #     bar2 = plt.bar(classes, FP, bottom=TP, label="FP", color="red")
    #     bar3 = plt.bar(classes, FN, bottom=TP + FP, label="FN", color="orange")
    #
    #     for bars in [bar1, bar2, bar3]:
    #         for bar in bars:
    #             height = bar.get_height()
    #             if height > 0:
    #                 plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2,
    #                          f"{int(height)}", ha='center', va='center', fontsize=8, color="white")
    #
    #     plt.ylabel("Count")
    #     plt.title("TP / FP / FN per Class")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.inference_dir, "confusion_stats_stacked11111.png"))
    #     plt.close()

    def plot_canonical_confusion_matrix(self):
        """
        Proper object detection confusion matrix using 1-to-1 IoU-based matching.
        - GT vs Pred, including class confusions and background FP/FN.
        - (N+1)x(N+1) shape: last row/col = unmatched FP/FN
        """
        df = self.df.copy()
        num_classes = len(self.class_names)
        bg_idx = num_classes
        label_names = self.class_names + ["background"]

        # Initialize pair storage
        y_true, y_pred = [], []

        for img_id in df["image"].unique():
            df_img = df[df["image"] == img_id]

            gt_boxes = df_img[df_img["gt_class"] != -1].copy()
            pred_boxes = df_img[df_img["pred_class"] != -1].copy()

            matched_gt_idx = set()
            matched_pred_idx = set()

            # Step 1: Match preds to GT by IoU ≥ 0.5 (1-to-1, max IoU first)
            for pred_idx, pred_row in pred_boxes.iterrows():
                # Get all valid GTs with IoU ≥ threshold
                ious = gt_boxes["iou"]
                valid_matches = gt_boxes[(ious >= 0.5) & (~gt_boxes.index.isin(matched_gt_idx))]

                if valid_matches.empty:
                    # No GT match → FP
                    y_true.append(bg_idx)
                    y_pred.append(pred_row["pred_class"])
                    continue

                # Get best IoU GT
                best_gt = valid_matches.loc[valid_matches["iou"].idxmax()]
                gt_idx = best_gt.name

                # Log the match
                gt_cls = int(best_gt["gt_class"])
                pred_cls = int(pred_row["pred_class"])

                y_true.append(gt_cls)
                y_pred.append(pred_cls)

                matched_gt_idx.add(gt_idx)
                matched_pred_idx.add(pred_idx)

            # Step 2: Remaining unmatched GTs → FN
            for gt_idx, gt_row in gt_boxes.iterrows():
                if gt_idx not in matched_gt_idx:
                    y_true.append(int(gt_row["gt_class"]))
                    y_pred.append(bg_idx)

            # Step 3: Remaining unmatched predictions → FP
            for pred_idx, pred_row in pred_boxes.iterrows():
                if pred_idx not in matched_pred_idx:
                    y_true.append(bg_idx)
                    y_pred.append(int(pred_row["pred_class"]))

        # Create confusion matrix
        labels = list(range(num_classes)) + [bg_idx]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(xticks_rotation=45, cmap="Purples", values_format="d")
        plt.title("Full Confusion Matrix \n IoU ≥ 0.5, Conf ≥ 0.6")
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "confusion_matrix_canonical.png"), dpi=300)
        plt.close()

        # Save CSV
        df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
        df_cm.to_csv(os.path.join(self.inference_dir, "confusion_matrix_canonical.csv"))

        # Accuracy = trace / total (excluding background row/col)
        core_cm = cm[:num_classes, :num_classes]
        accuracy = np.trace(core_cm) / np.sum(core_cm) if np.sum(core_cm) > 0 else 0
        with open(os.path.join(self.inference_dir, "accuracy.txt"), "w") as f:
            f.write("Canonical Confusion Matrix with Background and Class Confusions\n")
            f.write("IoU ≥ 0.5 | Conf ≥ 0.6\n")
            f.write(f"Overall Accuracy (no BG): {accuracy:.4f}\n")

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
            FP = df_cm.iloc[-1, i]  # background → class i
            FN = df_cm.iloc[i, -1]  # class i → background

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            acc = TP / df_cm.iloc[i, :-1].sum() if df_cm.iloc[i, :-1].sum() > 0 else 0

            results.append({
                "Class": self.class_names[i],
                "TP": int(TP),
                "FP": int(FP),
                "FN": int(FN),
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1 Score": round(f1, 3),
                "Accuracy": round(acc, 3)
            })

        df_results = pd.DataFrame(results)

        # ----- Add model-level summary -----

        total_TP = sum([row["TP"] for row in results])
        total_FP = sum([row["FP"] for row in results])
        total_FN = sum([row["FN"] for row in results])
        total_GT = total_TP + total_FN

        # Micro metrics (global TP / global FP/FN)
        micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                    micro_precision + micro_recall) > 0 else 0
        micro_accuracy = total_TP / total_GT if total_GT > 0 else 0

        # Macro metrics (mean across classes)
        macro_precision = df_results["Precision"].mean()
        macro_recall = df_results["Recall"].mean()
        macro_f1 = df_results["F1 Score"].mean()
        macro_accuracy = df_results["Accuracy"].mean()

        # Append model summary row
        summary_row = {
            "Class": "Model Total",
            "TP": total_TP,
            "FP": total_FP,
            "FN": total_FN,
            "Precision": round(micro_precision, 3),
            "Recall": round(micro_recall, 3),
            "F1 Score": round(micro_f1, 3),
            "Accuracy": round(micro_accuracy, 3)
        }
        df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)
        ###############################################################
        plt.figure(figsize=(10, len(df_results) * 0.6))
        sns.heatmap(df_results.set_index("Class"), annot=True, fmt=".3f", cmap="Blues", cbar=False, linewidths=0.5)
        plt.title("Precision / Recall / F1 / Accuracy per Class", fontsize=14)
        plt.tight_layout()
        # plt.savefig(os.path.join(self.inference_dir, "precision_recall_f1_table.png"), dpi=300)
        # plt.close()
        plt.figtext(0.5, -0.03,
                    "\n\n  Note: Class-level Accuracy is TP / class total (ignores inter-class confusions).\n"
                    "Model Total reflects overall detection rate across all classes.",
                    wrap=True, horizontalalignment='center', fontsize=9, color='gray')
        plt.savefig(os.path.join(self.inference_dir, "precision_recall_f1_table.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def extract_misclassified_images(self):
        df = self.df.copy()
        df["match_type"] = df.apply(lambda row: (
            "TP" if row["gt_class"] == row["pred_class"] and row["gt_class"] != -1 else
            "FP" if row["gt_class"] == -1 else
            "FN"
        ), axis=1)
        os.makedirs(self.vis_errors_out, exist_ok=True)
        misclassified = df[df["match_type"] != "TP"]
        for filename in misclassified["image"].unique():
            src = os.path.join(self.vis_folder, filename)
            dst = os.path.join(self.vis_errors_out, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)

    def plot_iou_distribution(self):
        df = self.df[self.df["iou"] > 0].copy()
        df["match_type"] = df.apply(lambda row: (
            "TP" if row["gt_class"] == row["pred_class"] and row["gt_class"] != -1 else
            "FP" if row["gt_class"] == -1 else
            "FN"
        ), axis=1)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="pred_class", y="iou", hue="match_type")
        plt.title("IoU Distribution by Predicted Class and Match Type")
        plt.xlabel("Predicted Class")
        plt.ylabel("IoU")
        plt.xticks(ticks=range(len(self.class_names)), labels=self.class_names)
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "iou_distribution.png"), dpi=300)
        plt.close()

    '''To Delete if not needed'''

    def plot_iou_mean_bar(self):
        df = self.df[self.df["iou"] > 0]
        mean_ious = df[df["gt_class"] != -1].groupby("gt_class")["iou"].mean()

        valid_indexes = [i for i in mean_ious.index if 0 <= i < len(self.class_names)]
        valid_class_names = [self.class_names[i] for i in valid_indexes]
        valid_ious = [mean_ious[i] for i in valid_indexes]

        plt.figure(figsize=(8, 5))
        plt.bar(valid_class_names, valid_ious, color="skyblue")
        plt.title("Mean IoU per Class")
        plt.xlabel("Class")
        plt.ylabel("Mean IoU")
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "mean_iou_per_class.png"), dpi=300)
        plt.close()

    def plot_f1_vs_threshold(self):
        thresholds = np.linspace(0.1, 0.9, 9)
        f1_results = []
        for t in thresholds:
            df_thresh = self.df[self.df['iou'] >= t].copy()
            df_thresh["match_type"] = df_thresh.apply(lambda row: (
                "TP" if row["gt_class"] == row["pred_class"] and row["gt_class"] != -1 else
                "FP" if row["gt_class"] == -1 else
                "FN"
            ), axis=1)
            for cls_id in range(len(self.class_names)):
                TP = len(df_thresh[(df_thresh["gt_class"] == cls_id) & (df_thresh["match_type"] == "TP")])
                FP = len(df_thresh[(df_thresh["pred_class"] == cls_id) & (df_thresh["match_type"] == "FP")])
                FN = len(df_thresh[(df_thresh["gt_class"] == cls_id) & (df_thresh["match_type"] == "FN")])
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_results.append(
                    {"class_id": cls_id, "class_name": self.class_names[cls_id], "threshold": t, "F1": f1})

        df_f1 = pd.DataFrame(f1_results)
        plt.figure(figsize=(10, 6))
        for cls_id in range(len(self.class_names)):
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

    def plot_max_f1_summary_table(self):
        thresholds = [round(x, 2) for x in np.linspace(0.1, 0.9, 9)]
        best_results = []
        for cls_id in range(len(self.class_names)):
            best_f1, best_threshold, best_tp, best_fp, best_fn = 0, None, 0, 0, 0
            for t in thresholds:
                df_thresh = self.df[self.df["iou"] >= t].copy()
                df_thresh["match_type"] = df_thresh.apply(lambda row: (
                    "TP" if row["gt_class"] == row["pred_class"] and row["gt_class"] != -1 else
                    "FP" if row["gt_class"] == -1 else
                    "FN"
                ), axis=1)
                TP = len(df_thresh[(df_thresh["gt_class"] == cls_id) & (df_thresh["match_type"] == "TP")])
                FP = len(df_thresh[(df_thresh["pred_class"] == cls_id) & (df_thresh["match_type"] == "FP")])
                FN = len(df_thresh[(df_thresh["gt_class"] == cls_id) & (df_thresh["match_type"] == "FN")])
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                if f1 > best_f1:
                    best_f1, best_threshold, best_tp, best_fp, best_fn = f1, t, TP, FP, FN
            best_results.append(
                {"class_id": cls_id, "class_name": self.class_names[cls_id], "max_F1": round(best_f1, 4),
                 "threshold": best_threshold, "TP": best_tp, "FP": best_fp, "FN": best_fn})

        df_summary = pd.DataFrame(best_results)
        df_summary.to_csv(os.path.join(self.inference_dir, "max_f1_summary_table.csv"), index=False)

    def plot_max_f1_summary_heatmap(self):
        path = os.path.join(self.inference_dir, "max_f1_summary_table.csv")
        if not os.path.exists(path):
            print("❌ Cannot find max_f1_summary_table.csv. Please run plot_max_f1_summary_table() first.")
            self.plot_max_f1_summary_table()
            return
        df = pd.read_csv(path).set_index("class_name")[["max_F1", "threshold", "TP", "FP", "FN"]]
        plt.figure(figsize=(10, 3.5))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor='gray')
        plt.title("Max F1 Summary (per Class)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "max_f1_summary_table.png"), dpi=300)
        plt.close()

    # def classify_match(row):
    #     if row["gt_class"] == -1:
    #         return "FP"
    #     elif row["pred_class"] == -1:
    #         return "FN"
    #     elif row["gt_class"] == row["pred_class"]:
    #         return "TP"
    #     else:
    #         return "Wrong"
    #
    # df["match_type"] = df.apply(classify_match, axis=1)

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
            FP = df_cm.iloc[-1, i]  # background → class i (unmatched predictions)
            FN = df_cm.iloc[i, -1]  # class i → background (missed detections)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            stats.append({
                "class_id": i,
                "class_name": self.class_names[i],
                "TP": TP, "FP": FP, "FN": FN,
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1": round(f1, 3)
            })

        df_stats = pd.DataFrame(stats)
        df_stats.to_csv(os.path.join(self.inference_dir, "precision_recall_per_class.csv"), index=False)

        # Bar plot values
        x = np.arange(len(self.class_names))
        width = 0.20
        fig, ax = plt.subplots(figsize=(10, 5))

        precisions = df_stats["Precision"].values
        recalls = df_stats["Recall"].values
        f1s = df_stats["F1"].values

        bars1 = ax.bar(x - width, precisions, width, label="Precision", color="deepskyblue")
        bars2 = ax.bar(x, recalls, width, label="Recall", color="lightcoral")
        bars3 = ax.bar(x + width, f1s, width, label="F1 Score", color="mediumseagreen")

        # Annotate bars
        for i in range(len(self.class_names)):
            ax.text(x[i] - width, precisions[i] + 0.01, f"{precisions[i]:.3f}", ha='center', fontsize=8)
            ax.text(x[i], recalls[i] + 0.01, f"{recalls[i]:.3f}", ha='center', fontsize=8)
            ax.text(x[i] + width, f1s[i] + 0.01, f"{f1s[i]:.3f}", ha='center', fontsize=8)

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
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import matplotlib.pyplot as plt
        import matplotlib.pyplot as plt

        df = self.df.copy()
        class_names = self.class_names

        df = df[df["pred_class"] != -1].copy()
        df = df[df["score"].notnull()]

        plt.figure(figsize=(10, 6))

        for cls_id, cls_name in enumerate(class_names):
            y_true = (df["gt_class"] == cls_id).astype(int)
            # y_score = df[df["pred_class"] == cls_id]["score"].reindex(df.index, fill_value=0)
            y_score = df["score"].where(df["pred_class"] == cls_id, 0)

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

    def plot_precision_recall_vs_threshold(self):
        df = self.df.copy()
        thresholds = np.linspace(0.0, 1.0, 101)
        precisions, recalls, f1s = [], [], []

        for t in thresholds:
            df_thresh = df[df["score"] >= t]

            TP = len(df_thresh[
                         (df_thresh["gt_class"] == df_thresh["pred_class"]) &
                         (df_thresh["gt_class"] != -1) &
                         (df_thresh["iou"] >= 0.5)
                         ])

            FP = len(df_thresh[df_thresh["gt_class"] == -1])

            # Instead of re-using pred_class == -1, treat low-score predictions as removed
            matched_gt_ids = df_thresh[
                (df_thresh["gt_class"] == df_thresh["pred_class"]) &
                (df_thresh["gt_class"] != -1) &
                (df_thresh["iou"] >= 0.5)
                ]["gt_class"].count()  # count of actual matched GTs at this threshold

            total_gt = len(df[df["gt_class"] != -1])
            FN = total_gt - matched_gt_ids

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, precisions, label="Precision", color="blue")
        plt.plot(thresholds, recalls, label="Recall", color="orange")
        plt.plot(thresholds, f1s, label="F1 Score", color="green")

        # Highlight best F1
        best_f1 = max(f1s)
        best_idx = f1s.index(best_f1)
        best_threshold = thresholds[best_idx]
        plt.axvline(x=best_threshold, color="gray", linestyle="--", linewidth=1)
        plt.text(best_threshold + 0.01, best_f1, f"Best F1 = {best_f1:.3f}", color="green", fontsize=9)

        plt.title("Precision / Recall / F1 vs Threshold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "pr_f1_vs_threshold.png"), dpi=300)
        plt.close()

    def plot_coco_metrics_table(self):
        import json
        import seaborn as sns
        import matplotlib.pyplot as plt

        path = os.path.join(self.inference_dir, "metrics.json")
        if not os.path.exists(path):
            print("❌ COCO metrics.json not found.")
            return

        if os.path.getsize(path) == 0:
            print(f"❌ metrics.json is empty: {path}")
            return

        with open(path, "r") as f:
            content = f.read().strip()
            if not content:
                print(f"❌ metrics.json has no content: {path}")
                return
            try:
                raw_data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                return

        bbox_data = raw_data.get("bbox", {})
        if not bbox_data:
            print(f"❌ No bbox section found in metrics.json at {path}")
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
            print(f"❌ Empty metrics table for {path}")
            return

        plt.figure(figsize=(7, len(df) * 0.4))
        sns.heatmap(df.set_index("Metric"), annot=True, fmt=".3f", cmap="YlGnBu", cbar=False, linewidths=0.5)
        plt.title("COCO Evaluation Metrics (bbox)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "coco_metrics_table.png"), dpi=300)
        plt.close()

    def plot_binary_class_metrics(self):
        if "roc_auc" not in self.df.columns or "tnr" not in self.df.columns:
            print("❌ ROC-AUC or TNR columns not found in prediction log.")
            return

        # Use only first row values (same for all)
        roc_auc = self.df["roc_auc"].iloc[0]
        tnr = self.df["tnr"].iloc[0]

        summary = pd.DataFrame({
            "Metric": ["ROC-AUC", "TNR"],
            "Value": [roc_auc, tnr]
        })
        summary.to_csv(os.path.join(self.inference_dir, "binary_metrics_summary.csv"), index=False)

        # Plot
        plt.figure(figsize=(5, 4))
        sns.barplot(data=summary, x="Metric", y="Value", hue="Metric", palette="Set2", legend=False)
        plt.title("Binary Classification Metrics")
        plt.ylim(0, 1)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        for i, row in summary.iterrows():
            plt.text(i, row["Value"] + 0.02, f"{row['Value']:.3f}", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_dir, "binary_metrics_summary.png"), dpi=300)
        plt.close()
        print("✅ Saved binary classification ROC-AUC and TNR plot.")

    def export_pdf_report(self):
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg

        model_name = os.path.basename(self.inference_dir.rstrip("/"))
        pdf_path = os.path.join(self.inference_dir, f"evaluation_summary_{model_name}.pdf")
        plots = [
            "confusion_matrix_canonical.png",  # High-res, correct logic
            "confusion_stats_stacked.png",  # Good for class-wise FP/FN pattern
            "precision_recall_per_class.png",  # Summary per class,
            "precision_recall_f1_table.png",  # Detailed table with all core metrics
            "precision_recall_per_class_curves.png",
            # "pr_f1_vs_threshold.png",
            "coco_metrics_table.png",
            "binary_metrics_summary.png",
            # "per_class_ap_summary.png",
            # "mean_iou_per_class.png",
            # "confusion_stats_grouped.png",  #  Helps ESA visualize breakdown

            # "confusion_matrix.png",
            # "confusion_matrix_full.png",
            # "confusion_stats_grouped.png",
            # "f1_score_per_class.png",
            # "max_f1_summary_table.png"
        ]

        with PdfPages(pdf_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            plt.axis("off")
            plt.text(0.5, 0.6, "Evaluation Report", ha="center", va="center", fontsize=24, weight='bold')
            plt.text(0.5, 0.45, f"Model: {model_name}", ha="center", va="center", fontsize=16)
            plt.text(0.5, 0.3, f"Path: {self.inference_dir}", ha="center", va="center", fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

            for plot in plots:
                path = os.path.join(self.inference_dir, plot)
                if os.path.exists(path):
                    fig = plt.figure(figsize=(12, 7))  # Increased resolution space
                    img = mpimg.imread(path)
                    plt.imshow(img, interpolation="none")
                    plt.axis("off")
                    # plt.title(plot.replace("_", " ").replace(".png", "").title(), fontsize=14)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

        print(f"✅ Evaluation summary PDF saved to: {pdf_path}")


def batch_run_all(base_dir, skip_visuals=False, skip_plots=False):
    print(f" Running batch evaluation in: {base_dir}\n")
    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue
        csv_path = os.path.join(full_path, "prediction_log.csv")
        if not os.path.exists(csv_path):
            continue
        print(f"📁 Evaluating folder: {folder}")
        try:
            evaluator = InfraEvaluator(full_path, skip_visuals=skip_visuals, skip_plots=skip_plots)
            evaluator.run_all()
            evaluator.export_pdf_report()
        except Exception as e:
            print(f"❌ Failed in {folder}: {e}")
    print("\n✅ Batch evaluation completed.")


# Single folder
# evaluator = InfraEvaluator("./Inference_result/Infra_sweep_lr0.0002_b512/", skip_visuals=False, skip_plots=False)
# evaluator.run_all()

# Batch mode
batch_run_all("/media/gpaps/My Passport/CVRL-GeorgeP/Trained_models/ShipOpt/run_1/fine_tune_lr15_b512_/", skip_visuals=True, skip_plots=False)
# batch_run_all("./Inference_result/Optical/KCP3_/res50_256/", skip_visuals=True, skip_plots=False)
# batch_run_all("./Inference_result/Optical/run_0/bestmodel_/", skip_visuals=True, skip_plots=False)
# batch_run_all("./Inference_result/Optical/SkySAT_boostAP_/bestmodel_/", skip_visuals=True, skip_plots=False)
# batch_run_all("./Inference_result/SAR/singleRun_run_0/bestmodel_/", skip_visuals=True, skip_plots=False)
# batch_run_all("./Inference_result/SAR/singleRun_run_0/bestmodel_/", skip_visuals=True, skip_plots=False)
