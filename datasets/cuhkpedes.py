import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class CUHKPEDES(BaseDataset):
    """
    CUHK-PEDES

    Reference:
    Person Search With Natural Language Description (CVPR 2017)

    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Person_Search_With_CVPR_2017_paper.html

    Dataset statistics:
    ### identities: 13003
    ### images: 40206,  (train)  (test)  (val)
    ### captions: 
    ### 9 images have more than 2 captions
    ### 4 identity have only one image

    annotation format: 
    [{'split', str,
      'captions', list,
      'file_path', str,
      'processed_tokens', list,
      'id', int}...]
    """
    dataset_dir = 'CUHK-PEDES'

    def __init__(self, root='', verbose=True):
        super(CUHKPEDES, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        self.anno_path = op.join(self.dataset_dir, 'reid_raw.json')
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        # Verify expected statistics
        self._verify_splits()

        if verbose:
            self.logger.info("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()

    def _verify_splits(self):
        """Hard assertions for expected data splits."""
        # Expected counts
        assert len(self.train_annos) == 34054, f"Train images: expected 34054, got {len(self.train_annos)}"
        assert len(self.val_annos) == 3078, f"Val images: expected 3078, got {len(self.val_annos)}"
        assert len(self.test_annos) == 3074, f"Test images: expected 3074, got {len(self.test_annos)}"

        # Expected unique IDs
        assert len(self.train_id_container) == 11003, f"Train unique ids: expected 11003, got {len(self.train_id_container)}"
        assert len(self.val_id_container) == 1000, f"Val unique ids: expected 1000, got {len(self.val_id_container)}"
        assert len(self.test_id_container) == 1000, f"Test unique ids: expected 1000, got {len(self.test_id_container)}"

        # ID sets must be disjoint
        train_ids = self.train_id_container
        val_ids = self.val_id_container
        test_ids = self.test_id_container

        assert len(train_ids & val_ids) == 0, f"Train and Val id sets intersect: {len(train_ids & val_ids)} ids"
        assert len(train_ids & test_ids) == 0, f"Train and Test id sets intersect: {len(train_ids & test_ids)} ids"
        assert len(val_ids & test_ids) == 0, f"Val and Test id sets intersect: {len(val_ids & test_ids)} ids"


    def _split_anno(self, anno_path: str):
        """Split annotations by the 'split' field in reid_raw.json.

        - split=='train' -> train_annos
        - split=='test' -> test_annos
        - split=='val' -> val_annos

        No random splitting. Data划分唯一来源是 reid_raw.json 的 'split' 字段。
        """
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            elif anno['split'] == 'val':
                val_annos.append(anno)
            else:
                raise ValueError(f"Unknown split value: {anno['split']}")
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):
        """Process annotations.

        Training: remap pid to 0-indexed consecutive integers via pid2label dict.
                  Each (image, caption) pair expands to one training sample.
        Val/Test: keep original pid (no remapping). Create four parallel lists:
                  image_pids/img_paths (one per image) and
                  caption_pids/captions (one per caption).
                  Gallery and query are independent lists, NOT paired.
        """
        if training:
            # Build pid2label: map original pid (int(anno['id']) - 1) to 0-indexed consecutive
            original_pids = set()
            for anno in annos:
                pid = int(anno['id']) - 1
                original_pids.add(pid)

            sorted_pids = sorted(original_pids)
            pid2label = {pid: idx for idx, pid in enumerate(sorted_pids)}

            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id']) - 1
                pid = pid2label[pid]  # remap to 0-indexed consecutive
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions']
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1

            # Assertions for training set
            assert len(dataset) > 0, "Training set is empty"
            assert len(original_pids) > 0, "No unique IDs found in training set"

            return dataset, set(range(len(sorted_pids)))
        else:
            # Val/Test: keep original pid, no remapping
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            pid_container = set()

            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions']
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)

            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
