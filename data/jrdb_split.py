import glob
import os
import ipdb

def get_jackrabbot_split(data_root):
     split_data = []
     for split in ['train', 'val', 'test']:
          files = sorted(glob.glob(f'{data_root}/label/{split}/*.txt'))
          scenes = [os.path.splitext(os.path.basename(x))[0] for x in files]
          split_data.append(scenes)
     return split_data