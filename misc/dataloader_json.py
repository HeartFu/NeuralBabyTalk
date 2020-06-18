import os

import json
from torch.utils.data import Dataset


class JSONShardDataset(Dataset):
    def __init__(self, shard_dir, shard_names=None, primary_key=None, stride=1):
        super().__init__()
        self.shard_dir = shard_dir
        self.shard_names = shard_names
        with open(os.path.join(self.shard_dir, self.shard_names[0]), 'r') as f:
            data = json.load(f)
        self.proposals = data


    def __len__(self):
        return len(self.proposals)

    def __getitem__(self, img_id):
        return self.proposals[str(img_id)]

class JSONSingleDataset(JSONShardDataset):
    def __init__(self, json_path, primary_key=None, stride=1):
        super().__init__(
            os.path.dirname(json_path),
            shard_names=[os.path.basename(json_path)],
            primary_key=primary_key,
            stride=stride,
        )
