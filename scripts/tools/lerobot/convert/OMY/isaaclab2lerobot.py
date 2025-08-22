import os
import h5py
import numpy as np

from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
"""
NOTE: Please use the environment of lerobot.

Because lerobot is rapidly developing, we don't guarantee the compatibility for the latest version of lerobot.
Currently, the commit we used is https://github.com/huggingface/lerobot/commit/26cb4614c961e6da04e4b83b6178331f4150650d
"""

# Feature definition for single-arm omy_follower
ARM_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (10,),
        "names": [
            "joint1.pos",
            "joint2.pos",
            "joint3.pos",
            "joint4.pos",
            "joint5.pos",
            "joint6.pos",
            "rh_l1.pos",
            "rh_l2.pos",
            "rh_r1_joint.pos",
            "rh_r2.pos",
        ]
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (10,),
        "names": [
            "joint1.pos",
            "joint2.pos",
            "joint3.pos",
            "joint4.pos",
            "joint5.pos",
            "joint6.pos",
            "rh_l1.pos",
            "rh_l2.pos",
            "rh_r1_joint.pos",
            "rh_r2.pos",
        ]

    },
    # "observation.images.front": {
    #     "dtype": "video",
    #     "shape": [480, 640, 3],
    #     "names": ["height", "width", "channels"],
    #     "video_info": {
    #         "video.height": 480,
    #         "video.width": 640,
    #         "video.codec": "av1",
    #         "video.pix_fmt": "yuv420p",
    #         "video.is_depth_map": False,
    #         "video.fps": 30.0,
    #         "video.channels": 3,
    #         "has_audio": False,
    #     },
    # },
    # "observation.images.wrist": {
    #     "dtype": "video",
    #     "shape": [480, 640, 3],
    #     "names": ["height", "width", "channels"],
    #     "video_info": {
    #         "video.height": 480,
    #         "video.width": 640,
    #         "video.codec": "av1",
    #         "video.pix_fmt": "yuv420p",
    #         "video.is_depth_map": False,
    #         "video.fps": 30.0,
    #         "video.channels": 3,
    #         "has_audio": False,
    #     },
    # }
}


def process_single_arm_data(dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str) -> bool:
    try:
        actions = np.array(demo_group['obs/actions'])
        joint_pos = np.array(demo_group['obs/joint_pos'])
        # front_images = np.array(demo_group['obs/front'])
        # wrist_images = np.array(demo_group['obs/wrist'])
    except KeyError:
        print(f'Demo {demo_name} is not valid, skip it')
        return False


    assert actions.shape[0] == joint_pos.shape[0]
    total_state_frames = actions.shape[0]
    # skip the first 5 frames
    for frame_index in tqdm(range(5, total_state_frames), desc='Processing each frame'):
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
            # "observation.images.front": front_images[frame_index],
            # "observation.images.wrist": wrist_images[frame_index],
        }
        dataset.add_frame(frame=frame, task=task)

    return True

def convert_isaaclab_to_lerobot():
    """NOTE: Modify the following parameters to fit your own dataset"""
    repo_id = 'data/omy_stack'
    robot_type = 'omy_follower'
    fps = 30
    hdf5_root = './datasets'
    hdf5_files = [os.path.join(hdf5_root, 'dataset.hdf5')]
    task = 'OMY_Stack'
    push_to_hub = False

    """convert to LeRobotDataset"""
    now_episode_index = 0
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=ARM_FEATURES,
    )

    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f'[{hdf5_id+1}/{len(hdf5_files)}] Processing hdf5 file: {hdf5_file}')
        with h5py.File(hdf5_file, 'r') as f:
            demo_names = list(f['data'].keys())
            print(f'Found {len(demo_names)} demos: {demo_names}')

            for demo_name in tqdm(demo_names, desc='Processing each demo'):
                demo_group = f['data'][demo_name]
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f'Demo {demo_name} is not successful, skip it')
                    continue

                valid = process_single_arm_data(dataset, task, demo_group, demo_name)

                if valid:
                    now_episode_index += 1
                    dataset.save_episode()
                    print(f'Saving episode {now_episode_index} successfully')

    if push_to_hub:
        dataset.push_to_hub()

if __name__ == '__main__':
    convert_isaaclab_to_lerobot()
