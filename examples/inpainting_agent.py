# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mbodied.agents.sense.sensory_agent import SensoryAgent
from mbodied.types.sense.vision import Image


class InpaintAgent(SensoryAgent):
    """Augment agent to generate augmented images using i.e. stable diffusion."""

    def __init__(
        self,
        model_src="http://3.236.52.5:7860/",
        model_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            model_src=model_src,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    def act(self, detection_prompt: str, inpaint_prompt: str, image: Image, *args, **kargs) -> Image:
        """Act based on the prompt and image using the remote augment server.

        Args:
            instruction (str): The instruction to act on.
            image (Image): The image to act on.
            *args: Variable length argument list.
            **kargs: Arbitrary keyword arguments.

        Returns:
            Image: The augmented image generated by the agent
        """
        if self.actor is None:
            raise ValueError("Remote actor for Augment agent not initialized.")
        tmp_path = "temp.jpg"
        image.save(tmp_path)
        response, img = self.actor.predict(
            tmp_path,
            detection_prompt,  # str  in 'Detection Prompt[To detect multiple objects, seperating each with '.', like this: cat . dog . chair ]' Textbox component
            "inpainting",  # str  in 'Task type' Radio component
            inpaint_prompt,  # str  in 'Inpaint/Outpaint Prompt (if this is empty, then remove)' Textbox component
            0.3,  # int | float (numeric value between 0.0 and 1.0) in 'Box Threshold' Slider component
            0.25,  # int | float (numeric value between 0.0 and 1.0) in 'Text Threshold' Slider component
            0.8,  # int | float (numeric value between 0.0 and 1.0) in 'IOU Threshold' Slider component
            "merge",  # str  in 'inpaint_mode' Radio component
            "type what to detect below",  # str  in 'Mask from' Radio component
            "segment",  # str  in 'remove mode' Radio component
            "10",  # str  in 'remove_mask_extend' Textbox component
            5,  # int | float (numeric value between 1 and 20) in 'How many relations do you want to see' Slider component
            "Brief",  # str  in 'Kosmos Description Type' Radio component
            fn_index=2,
        )
        return Image(img)


# Example usage:
if __name__ == "__main__":
    augment_agent = InpaintAgent()
    result = augment_agent.act(
        detection_prompt="white wall", inpaint_prompt="window", image=Image("assets/arm.jpg")
    )
    print(result)
    result.pil.show()
