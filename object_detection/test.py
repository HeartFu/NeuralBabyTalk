import json

with open('../data/coco/obj_det_result_val.json', 'r') as f:
    data = json.load(f)

with open('../data/coco/nbt_obj_det_result_val.json', 'r') as f:
    data2 = json.load(f)

with open('../data/coco/annotations/instances_val2014.json', 'r') as f:
    anns = json.load(f)

print(data)