import os
import numpy as np
import cv2
import random


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicAugmentation():

    def __init__(self,
                 img_dir,
                 images,
                 coco,
                 input_dim=(160, 160),
                 preproc=None):

        super().__init__()
        self._input_dim = input_dim
        self.img_dir = img_dir
        self.images = images
        self.coco = coco

    def pull_item(self, index):
        bbs = []
        try:
            index = min(index, len(self.images) - 1)
            img_id = self.images[index]['id']
        except KeyError:
            __import__('pdb').set_trace()
            pass
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), 70)

        img = cv2.imread(img_path)
        anns_out = []
        for k in range(num_objs):
            ann = anns[k]
            anns_out.append(ann)
            bbox = ann['bbox']
            xmin, ymin, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
            bbs.append([xmin, ymin, width, height])
        return img, np.asarray(bbs), anns_out

    def __getitem__(self, idx):
        mosaic_labels = []
        input_dim = self._input_dim
        input_h, input_w = input_dim[0], input_dim[1]

        # yc, xc = s, s  # mosaic center x, y
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

        # 3 additional image indices
        indices = [idx] + \
            [random.randint(0, len(self.images) - 1) for _ in range(3)]

        anns_out = []
        for i_mosaic, index in enumerate(indices):
            img, _labels, anns = self.pull_item(index)
            anns_out.extend(anns)
            h0, w0 = img.shape[:2]  # orig hw
            scale = min(1. * input_h / h0, 1. * input_w / w0)
            img = cv2.resize(
                img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
            )
            # generate output mosaic image
            (h, w, c) = img.shape[:3]
            if i_mosaic == 0:
                mosaic_img = np.full(
                    (input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

            # suffix l means large image, while s means small image in mosaic aug.
            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
            )

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            labels = _labels.copy()
            # Normalized xywh to pixel xyxy format
            if _labels.size > 0:
                labels[:, 0] = scale * _labels[:, 0] + padw
                labels[:, 1] = scale * _labels[:, 1] + padh
                labels[:, 2] = scale * _labels[:, 2]  # + padw
                labels[:, 3] = scale * _labels[:, 3]  # + padh
                mosaic_labels.append(labels)

        if len(mosaic_labels):
            try:
                mosaic_labels = np.concatenate(mosaic_labels, 0)
            except ValueError:
                __import__('pdb').set_trace()
            np.clip(mosaic_labels[:, 0], 0, 2 *
                    input_w, out=mosaic_labels[:, 0])
            np.clip(mosaic_labels[:, 1], 0, 2 *
                    input_h, out=mosaic_labels[:, 1])
            np.clip(mosaic_labels[:, 2], 0, 2 *
                    input_w, out=mosaic_labels[:, 2])
            np.clip(mosaic_labels[:, 3], 0, 2 *
                    input_h, out=mosaic_labels[:, 3])

        if False:
            for box in mosaic_labels:
                color = (0, 255, 0)
                thickness = 2
                start_point = (int(box[0]), int(box[1]))
                end_point = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(mosaic_img, start_point, end_point, color, thickness)

            cv2.imwrite('mosaic_img.jpg', mosaic_img)

        img_info = {"file_name": 'input.jpg', "height": mosaic_img.shape[0], "width": mosaic_img.shape[1],
                    "id": 1}
        meta = dict(img=mosaic_img, img_info=img_info,
                    gt_bboxes=anns_out, gt_labels=mosaic_labels)
        return meta
