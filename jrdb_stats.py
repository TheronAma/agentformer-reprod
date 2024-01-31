import ipdb
import os
import json
import numpy as np
# CONSTANTS WE CHANGE

print('hi')
# root directory of jrdb dataset
jrdb_root = "jackrabbot"

output_dir = "stats"

# number of sequences to hold out of training for val
VAL_SEQUENCES = 0

# sequences we want to preprocess
locations = [
    "bytes-cafe-2019-02-07_0",
    "gates-ai-lab-2019-02-08_0",
    "gates-basement-elevators-2019-01-17_1",
    "hewlett-packard-intersection-2019-01-24_0",
    "huang-lane-2019-02-12_0",
    "jordan-hall-2019-04-22_0",
    "packard-poster-session-2019-03-20_1",
    "packard-poster-session-2019-03-20_2",
    "svl-meeting-gates-2-2019-04-08_0",
    "stlc-111-2019-04-19_0",
    "tressider-2019-03-16_0", 
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
    # 'clark-center-intersection-2019-02-28_0'
]
print('hi')

if __name__=="__main__":
    os.makedirs(output_dir, exist_ok=True)

    total_data = {}

    for location in locations:
        # add the location to the correct split
        f = open(f"{jrdb_root}/train/labels/labels_3d/{location}.json")

        location_data = json.load(f)

        labels = location_data["labels"]

        location_stats = {}

        frames = 0
        peds_in_frames = 0

        pedestrians = set()

        ped_trajs = {}

        for i, frame in enumerate(labels.keys()):
            curr_peds = set()

            for label in labels[frame]:
                ped_id = label['label_id']

                pedestrians.add(ped_id)
                curr_peds.add(ped_id)

                if ped_id not in ped_trajs:
                    ped_trajs[ped_id] = []
                
                ped_trajs[ped_id].append(np.array([label["box"]["cx"], label["box"]["cy"]]))

            peds_in_frames += len(curr_peds)
            frames += 1
        
        ped_speeds = {}

        for ped_id in ped_trajs.keys():
            traj = np.array(ped_trajs[ped_id])
            ped_trajs[ped_id] = traj

            vel_traj = traj[1:] - traj[:-1]
            speed_traj = np.linalg.norm(vel_traj, axis=-1)
            avg_speed = np.mean(speed_traj)

            ped_speeds[ped_id] = avg_speed

        avg_speed = 0
        for ped_id in ped_speeds.keys():
            avg_speed += ped_speeds[ped_id]
        avg_speed = avg_speed / len(ped_speeds)

        location_stats["ped_trajs"] = ped_trajs
        location_stats["total_peds"] = len(pedestrians)
        location_stats["avg_pedestrians"] = peds_in_frames / frames
        location_stats["avg_speed"] = avg_speed

        total_data[location] = location_stats
        
        # out_f = open(f'{output_dir}/{location}.txt', "a")
        # out_f.write("pedestrians: {}\n".format(location_stats["total_peds"]))
        # out_f.write("pedestrians per frame: {} \n".format(peds_in_frames / frames))
        # out_f.write()
        # out_f.close()
        f.close()
    
    out_f = open(f"{output_dir}/stats.csv", "a")
    out_f.write("location, total_pedestrians, avg_pedestrians_per_frame, avg_speed, total_sequences\n")

    for location in locations:
        stats = total_data[location]
        out_f.write(f"{location}, {stats['total_peds']}, {stats['avg_pedestrians']}, {stats['avg_speed']}\n")
    
    out_f.close()


    
    # for split in ["train", "val", "test"]:
    #     os.makedirs(f'{output_dir}/label/{split}', exist_ok=True)

    #     # jrdb_split = "test" if split == "test" else "train" 
        
    #     for location in location_splits[split]:
    #         print(location)
    #         f = open(f"{jrdb_root}/train/labels/labels_3d/{location}.json")

    #         location_data = json.load(f)

    #         labels = location_data["labels"]

    #         total_data = []

    #         for i, frame in enumerate(labels.keys()):
    #             if i % 6 == 1:
    #                 for label in labels[frame]:
    #                     data = np.ones(18) * -1.0
    #                     data[0] = i // 6
    #                     data[1] = int(label["label_id"].split(":")[1])
    #                     data[13] = label["box"]["cx"]
    #                     data[14] = label["box"]["cz"]
    #                     data[15] = label["box"]["cy"]
    #                     data[10] = label["box"]["l"]
    #                     data[11] = label["box"]["h"]
    #                     data[12] = label["box"]["w"]
    #                     data[16] = label["box"]["rot_z"]
    #                     data[17] = 1

    #                     data = data.astype(str)

    #                     data[2] = "Pedestrian"

    #                     total_data.append(data)

    #         total_data = np.stack(total_data)

    #         np.savetxt(f'{output_dir}/label/{split}/{location}.txt', total_data, fmt='%s')


            

            
    