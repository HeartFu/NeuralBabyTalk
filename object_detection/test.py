# import json
#
# with open('../data/coco/obj_det_result_val.json', 'r') as f:
#     data = json.load(f)
#
# with open('../data/coco/nbt_obj_det_result_val.json', 'r') as f:
#     data2 = json.load(f)
#
# with open('../data/coco/annotations/instances_val2014.json', 'r') as f:
#     anns = json.load(f)
#
# print(data)

# from pycocotools.coco import COCO
#
# coco = COCO('../data/coco/annotations/instances_val2014.json')
#
# print(coco.getCatIds())

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

print(len(COCO_INSTANCE_CATEGORY_NAMES))

a = {'a': 1, 'b':2, 'c': 3}

for k, v in a:
    print(k)
    print(v)

