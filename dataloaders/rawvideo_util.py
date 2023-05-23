import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
# pip install opencv-python
import cv2
import decord
from decord import VideoReader, cpu
decord.bridge.set_bridge("torch")

class RawVideoExtractorDecord():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, max_frames):
        try:
            vr = VideoReader(video_file, ctx=cpu(0))
            total_frame_num = len(vr)

            if max_frames < total_frame_num:
                sample_idx = np.linspace(0, total_frame_num-1, num=max_frames, dtype=int)
            else:
                sample_idx = np.arange(total_frame_num)

            images = vr.get_batch(sample_idx)  # (T, H, W, C)

            images = preprocess(images.permute(0, 3, 1, 2).float() / 255.)

            if images.shape[0] > 0:
                video_data = images
            else:
                video_data = th.zeros(1)
        except Exception as e:
            print("Error: ", e)
            video_data = th.zeros(1)
        return {'video': video_data}  # video_data shape here is (T,C,H,W)

    def get_video_data(self, video_path, max_frames):
        image_input = self.video_to_tensor(video_path, self.transform, max_frames)
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based Decord
RawVideoExtractor = RawVideoExtractorDecord