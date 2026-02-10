from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
import copy


class ClassAwareMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.cfg = cfg

        self.standard_aug = [
            T.ResizeShortestEdge(short_edge_length=(900, 1000, 1100, 1200), max_size=1333, sample_style="choice"),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomApply(T.RandomContrast(0.8, 1.2), prob=0.3),
            T.RandomApply(T.RandomBrightness(0.8, 1.2), prob=0.3),
            T.RandomApply(T.RandomSaturation(0.8, 1.2), prob=0.3),
        ]

        self.extra_aug = [
            T.RandomApply(T.RandomRotation(angle=[-45, 45]), prob=0.5),
        ]

        self.target_class_ids = [0, 2, 3, 4]

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        if not self.is_train:
            self.tfm_gens = [T.ResizeShortestEdge(short_edge_length=(1000, 1000), max_size=1333, sample_style="choice")]
            return super().__call__(dataset_dict)

        # Training Logic
        annos = dataset_dict.get("annotations", [])
        has_target = any(obj["category_id"] in self.target_class_ids for obj in annos)
        aug_list = self.standard_aug + self.extra_aug if has_target else self.standard_aug
        self.tfm_gens = aug_list

        try:
            return super().__call__(dataset_dict)
        except ValueError:
            return None