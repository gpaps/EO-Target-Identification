import random
from collections import defaultdict, Counter
from torch.utils.data import Sampler
from detectron2.data import MetadataCatalog
import os
import csv


class BalancedSampler(Sampler):
    """
    A custom sampler that enforces class-balanced sampling in each batch.

    Args:
        dataset (Dataset): The mapped dataset object.
        dataset_dicts (List[Dict]): Original dataset dictionaries with annotations.
        batch_size (int): Number of samples per batch.
        cfg (CfgNode): Detectron2 config object.
        oversample_factor (int): Controls how many samples per rare class are added to each batch.
            - A value of 1 means each class contributes one sample per batch (default).
            - Higher values (e.g., 2, 3) oversample underrepresented classes, increasing their exposure
              in training, which may improve recall but risks overfitting.
            - Should be tuned based on dataset imbalance and desired sensitivity to rare classes.
    """

    def __init__(self, dataset, dataset_dicts, batch_size, cfg, oversample_factor=1):
        self.dataset = dataset
        self.dataset_dicts = dataset_dicts
        self.batch_size = batch_size
        self.cfg = cfg
        self.oversample_factor = oversample_factor

        self.class_to_indices = defaultdict(list)
        for i, d in enumerate(self.dataset_dicts):
            classes = {anno["category_id"] for anno in d["annotations"]}
            for cls_id in classes:
                self.class_to_indices[cls_id].append(i)

        self.class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        self.counter = Counter()
        self.log_path = os.path.join(cfg.OUTPUT_DIR, "batch_class_distribution.csv")
        self._init_log_file()

    def _init_log_file(self):
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration"] + self.class_names)

    def _log_class_counts(self, step):
        row = [step] + [self.counter.get(i, 0) for i in range(len(self.class_names))]
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        self.counter.clear()

    def __iter__(self):
        step = 0
        rare_classes = self.cfg.RARE_CLASS_IDS if hasattr(self.cfg, "RARE_CLASS_IDS") else []

        while True:
            batch = []

            # 1. Add one sample per class
            for cls_id in self.class_to_indices:
                batch.append(random.choice(self.class_to_indices[cls_id]))

            # 2. Add oversamples for rare classes only
            for cls_id in rare_classes:
                for _ in range(self.oversample_factor - 1):
                    batch.append(random.choice(self.class_to_indices[cls_id]))

            # 3. Shuffle and trim to batch size
            random.shuffle(batch)
            batch = batch[:self.batch_size]

            # 4. Track class distribution
            for i in batch:
                classes = {anno["category_id"] for anno in self.dataset_dicts[i]["annotations"]}
                for cls in classes:
                    self.counter[cls] += 1

            self._log_class_counts(step)
            step += 1

            yield from batch

    def __len__(self):
        return 2 ** 30  # effectively infinite
