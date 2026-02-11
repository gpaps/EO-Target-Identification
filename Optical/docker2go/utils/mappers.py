import copy
import torch
import numpy as np
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


class ClassAwareMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.cfg = cfg

        # 1. STANDARD AUG
        self.standard_aug = [
            T.ResizeShortestEdge(short_edge_length=(800, 900, 1000, 1100, 1200), max_size=1333, sample_style="choice"),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomApply(T.RandomContrast(0.8, 1.2), prob=0.3),
            T.RandomApply(T.RandomBrightness(0.8, 1.2), prob=0.3),
            T.RandomApply(T.RandomSaturation(0.8, 1.2), prob=0.3),
        ]

        # 2. EXTRA AUG
        self.extra_aug = [
            T.RandomApply(T.RandomRotation(angle=[-45, 45]), prob=0.5),
        ]

        self.target_class_ids = [0, 2, 3, 4]
        self.tfm_gens = []

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        # 1. READ IMAGE
        image = utils.read_image(dataset_dict["file_name"], format=self.cfg.INPUT.FORMAT)
        utils.check_image_size(dataset_dict, image)

        # 2. AUGMENTATIONS
        if self.is_train:
            annos = dataset_dict.get("annotations", [])
            has_target = any(obj["category_id"] in self.target_class_ids for obj in annos)
            aug_list = self.standard_aug + self.extra_aug if has_target else self.standard_aug
            self.tfm_gens = aug_list
        else:
            self.tfm_gens = [T.ResizeShortestEdge(short_edge_length=(1000, 1000), max_size=1333, sample_style="choice")]

        # 3. APPLY TRANSFORMS
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]

        # --- FIX: Force float32 and copy to avoid 'not writable' warning ---
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)).copy()).float()
        # -----------------------------------------------------------------

        # 4. INSTANCES
        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.cfg.INPUT.MASK_FORMAT
            )

            if self.is_train:
                instances = utils.filter_empty_instances(instances)

            dataset_dict["instances"] = instances

        return dataset_dict