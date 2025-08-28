import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ENV_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (10,),
        "names": [
            "joint1.pos","joint2.pos","joint3.pos","joint4.pos","joint5.pos","joint6.pos",
            "rh_l1.pos","rh_l2.pos","rh_r1_joint.pos","rh_r2.pos",
        ]
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (10,),
        "names": [
            "joint1.pos","joint2.pos","joint3.pos","joint4.pos","joint5.pos","joint6.pos",
            "rh_l1.pos","rh_l2.pos","rh_r1_joint.pos","rh_r2.pos",
        ]
    },
    "observation.images.cam_wrist": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 224,"video.width": 224,"video.codec": "av1",
            "video.pix_fmt": "yuv420p","video.is_depth_map": False,
            "video.fps": 30.0,"video.channels": 3,"has_audio": False,
        },
    },
    "observation.images.cam_top": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 224,"video.width": 224,"video.codec": "av1",
            "video.pix_fmt": "yuv420p","video.is_depth_map": False,
            "video.fps": 30.0,"video.channels": 3,"has_audio": False,
        },
    }
}


def process_data(dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str) -> bool:
    try:
        actions = np.array(demo_group['obs/actions'])
        joint_pos = np.array(demo_group['obs/joint_pos'])
        cam_wrist_images = np.array(demo_group['obs/cam_wrist'])
        cam_top_images = np.array(demo_group['obs/cam_top'])
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
            "observation.images.cam_wrist": cam_wrist_images[frame_index],
            "observation.images.cam_top": cam_top_images[frame_index],
        }
        dataset.add_frame(frame=frame, task=task)

    return True


def convert_isaaclab_to_lerobot(task: str, repo_id: str, robot_type: str, dataset_path: str, fps: int, push_to_hub: bool = False):
    hdf5_files = [dataset_path]

    now_episode_index = 0
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=ENV_FEATURES,
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

                valid = process_data(dataset, task, demo_group, demo_name)

                if valid:
                    now_episode_index += 1
                    dataset.save_episode()
                    print(f'Saving episode {now_episode_index} successfully')

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot format")
    parser.add_argument("--task_name", type=str, required=True, help="Task name (e.g., OMY_Pickup)")
    parser.add_argument("--robot_type", type=str, default="aiworker",
        help="Robot type (default: aiworker)")
    parser.add_argument("--dataset_path", type=str, default="./datasets/dataset.hdf5",
        help="Path to dataset HDF5 file (default: ./datasets/dataset.hdf5)")
    parser.add_argument("--fps", type=int, default=30,
        help="Frames per second for dataset (default: 30)")
    parser.add_argument("--push_to_hub", action="store_true",
        help="Whether to push dataset to HuggingFace Hub (default: False)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    default_repo_id = f"data/{timestamp}"
    parser.add_argument("--repo_id", type=str, default=default_repo_id,
        help=f"Repo ID for saving dataset (default: {default_repo_id})")

    args = parser.parse_args()

    convert_isaaclab_to_lerobot(
        task=args.task_name,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        dataset_path=args.dataset_path,
        fps=args.fps,
        push_to_hub=args.push_to_hub
    )
