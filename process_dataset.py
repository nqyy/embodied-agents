import os
from mbodied.data.replaying import Replayer
import numpy as np

import os
from typing import Any, Callable, List, Tuple

import numpy as np
from datasets import Dataset, DatasetInfo, Features, Image, Value
from huggingface_hub import login

from mbodied.data.utils import infer_features


class CustomFolderReplayer:
    def __init__(self, path: str) -> None:
        """Initialize the FolderReplayer class with the given path.

        Args:
            path (str): Path to the folder containing HDF5 files.
        """
        self.path = path

    def compute_delta(self, current_action, last_action):
        """Compute the delta between the current and last action."""
        if isinstance(current_action, dict):
            delta = {}
            for key in current_action:
                delta[key] = self.compute_delta(current_action[key], last_action[key])
                # Check if the key is "grasp" and adjust the delta value accordingly
                if key == "grasp":
                    if delta[key] < -0.1:
                        delta[key] = -1
                    elif delta[key] > 0.1:
                        delta[key] = 1
                    else:
                        delta[key] = 0
            return delta
        else:
            return round(current_action - last_action, 5)

    def actions_equal(self, action1, action2):
        """Check if two actions are equal."""
        if isinstance(action1, dict) and isinstance(action2, dict):
            for key in action1:
                if not self.actions_equal(action1[key], action2[key]):
                    return False
            return True
        else:
            return action1 == action2

    def __iter__(self):
        """Iterate through the HDF5 files in the folder."""
        episode = 0
        for f in os.listdir(self.path):
            if f.endswith(".h5"):
                r = Replayer(f"{self.path}/{f}")
                last_action = None
                last_delta = None
                gripper_pos = 1
                for _i, sample in enumerate(r):
                    observation = sample[0]
                    action = sample[1]
                    image = np.asarray(observation["image"])
                    instruction = observation["instruction"]

                    if last_action is not None:
                        if self.actions_equal(action, last_action):
                            continue
                        action_delta = self.compute_delta(action, last_action)
                        if self.actions_equal(last_delta, action_delta):
                            continue
                        last_delta = action_delta
                        # Convert gripper to absolute value
                        grasp = gripper_pos + action_delta["grasp"]
                        if grasp < 0:
                            grasp = 0
                        if grasp > 1:
                            grasp = 1
                        action_delta["grasp"] = grasp
                        gripper_pos = grasp
                        print("delta:", action_delta)
                        yield {
                            "observation": {"image": image, "instruction": instruction},
                            "action": action_delta,
                            "episode": episode,
                        }

                    last_action = action
                episode += 1


def to_dataset(folder: str, name: str, description: str = None, **kwargs) -> None:
    """Convert the folder of HDF5 files to a Hugging Face dataset.

    Args:
        folder (str): Path to the folder containing HDF5 files.
        name (str): Name of the dataset.
        description (str, optional): Description of the dataset. Defaults to None.
        **kwargs: Additional arguments to pass to the Dataset.push_to_hub method.
    """
    r = CustomFolderReplayer(folder)
    data = list(r.__iter__())

    def list_of_dicts_to_dict(data: List[dict]) -> dict:
        if not data:
            return {}
        columnar_data = {key: [] for key in data[0]}
        for item in data:
            for key, value in item.items():
                columnar_data[key].append(value)
        return columnar_data

    features = Features(
        {
            "observation": {"image": Image(), "instruction": Value("string")},
            "action": infer_features(data[0]["action"]),
            "episode": Value("int32"),
        },
    )

    # Convert list of dicts to dict of lists, preserving order
    data_dict = {key: [item[key] for item in data] for key in data[0]}
    info = DatasetInfo(
        description=description,
        license="Apache-2.0",
        citation="None",
        size_in_bytes=8000000,
        features=features,
    )

    ds = Dataset.from_dict(data_dict, info=info)
    ds = ds.with_format("pandas")
    # login(os.getenv("HF_TOKEN"))
    # ds.push_to_hub(name, **kwargs)


def main():
    to_dataset("xarm_dataset_test", "mbodiai/xarm_7_6_delta", description="XArm dataset")


if __name__ == "__main__":
    main()
