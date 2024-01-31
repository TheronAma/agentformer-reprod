import ipdb
import os
import json
import numpy as np
# CONSTANTS WE CHANGE

# root directory of jrdb dataset
jrdb_root = "jackrabbot"

output_dir = "datasets/jackrabbot/"

# number of sequences to hold out of training for val
VAL_SEQUENCES = 0

# sequences we want to preprocess
locations = [
    "bytes-cafe-2019-02-07_0",
    'packard-poster-session-2019-03-20_1'
    "gates-ai-lab-2019-02-08_0",
    "gates-basement-elevators-2019-01-17_1",
    "hewlett-packard-intersection-2019-01-24_0",
    "huang-lane-2019-02-12_0",
    "jordan-hall-2019-04-22_0",
    "packard-poster-session-2019-03-20_1",
    "packard-poster-session-2019-03-20_2",
    "svl-meeting-gates-2-2019-04-08_0",
    "stlc-111-2019-04-19_0",
    "tressider-2019-03-16_0", # outdoors
    "svl-meeting-gates-2-2019-04-08_1",
    "tressider-2019-03-16_1",
]

# train and test portions of dataset
TRAIN = [
    'bytes-cafe-2019-02-07_0',
    'clark-center-2019-02-28_0',
    'clark-center-2019-02-28_1',
    
    'cubberly-auditorium-2019-04-22_0',
    'forbes-cafe-2019-01-22_0',
    'gates-159-group-meeting-2019-04-03_0',
    'gates-ai-lab-2019-02-08_0',
    'gates-basement-elevators-2019-01-17_1',
    'gates-to-clark-2019-02-28_1',
    'hewlett-packard-intersection-2019-01-24_0',
    'huang-2-2019-01-25_0',
    'huang-basement-2019-01-25_0',
    'huang-lane-2019-02-12_0',
    'jordan-hall-2019-04-22_0',
    'memorial-court-2019-03-16_0',
    'meyer-green-2019-03-16_0',
    'nvidia-aud-2019-04-18_0',
    'packard-poster-session-2019-03-20_0',
    
    'packard-poster-session-2019-03-20_2',
    'stlc-111-2019-04-19_0',
    
    'svl-meeting-gates-2-2019-04-08_1',
    'tressider-2019-03-16_0',
    
    'tressider-2019-04-26_2'
]


TEST = [
    'packard-poster-session-2019-03-20_1',
    'tressider-2019-03-16_1',
    'svl-meeting-gates-2-2019-04-08_0',
    'clark-center-intersection-2019-02-28_0'
]
if __name__=="__main__":
    location_splits = {}
    location_splits["train"] = []
    location_splits["val"] = []
    location_splits["test"] = []

    for location in locations:
        # add the location to the correct split
        if location in TRAIN:
            location_splits["train"].append(location)
        
        if location in TEST:
            location_splits["test"].append(location)
    
    for split in ["train", "val", "test"]:
        os.makedirs(f'{output_dir}/label/{split}', exist_ok=True)

        # jrdb_split = "test" if split == "test" else "train" 
        
        for location in location_splits[split]:
            print(location)
            f = open(f"{jrdb_root}/train/labels/labels_3d/{location}.json")

            location_data = json.load(f)

            labels = location_data["labels"]

            total_data = []

            for i, frame in enumerate(labels.keys()):
                if i % 6 == 1:
                    for label in labels[frame]:
                        data = np.ones(18) * -1.0
                        data[0] = i // 6
                        data[1] = int(label["label_id"].split(":")[1])
                        data[13] = label["box"]["cx"]
                        data[14] = label["box"]["cz"]
                        data[15] = label["box"]["cy"]
                        data[10] = label["box"]["l"]
                        data[11] = label["box"]["h"]
                        data[12] = label["box"]["w"]
                        data[16] = label["box"]["rot_z"]
                        data[17] = 1

                        data = data.astype(str)

                        data[2] = "Pedestrian"

                        total_data.append(data)

            total_data = np.stack(total_data)

            np.savetxt(f'{output_dir}/label/{split}/{location}.txt', total_data, fmt='%s')


            

            
    