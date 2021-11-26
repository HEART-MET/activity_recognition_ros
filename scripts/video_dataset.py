import torchvision.io as tio
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import numpy as np
import os
import glob
import cv2
import pdb

def make_dataset(root, labels_file):

    samples = {}
    samples['timestamps'] = []
    if not os.path.exists(root) or not os.path.exists(labels_file):
        return samples
    video_files = sorted(glob.glob(root + '/*.mp4'))

    labels = np.genfromtxt(labels_file, dtype=None, encoding=None)
    classes = {}
    for lbl in labels:
        classes[lbl[0]] = lbl[1]

    frames_per_video = []
    pts = []
    class_labels = []
    for vid in video_files:
        timestamps, fps = tio.read_video_timestamps(vid)
        timestamps = np.array(timestamps)
        timestamps = timestamps[timestamps >= 0]
        num_frames = len(timestamps)
        if os.path.basename(vid) not in classes.keys():
            print('Skipping %s since there is no label for it' % vid)
            continue
        frames_per_video.append(num_frames)
        pts.append(timestamps)
        class_labels.append(classes[os.path.basename(vid)])

    samples['timestamps'] = pts
    samples['num_frames_per_video'] = frames_per_video
    samples['labels'] = class_labels
    samples['video_paths'] = video_files

    return samples


class VideoDataset(data.Dataset):
    def __init__(self, root, labels, clip_length = 20, transform=None, num_classes=22):
        self.root = root
        samples = make_dataset(self.root, labels)
        if len(samples['timestamps']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)
        self.samples = samples
        self.frames_per_video = np.cumsum(self.samples['num_frames_per_video'])
        self.transform = transform
        self.clip_length = clip_length
        self.num_classes = num_classes

    def __getitem__(self, index):
        # get video id of requested frame
        video_idx = np.argmax(self.frames_per_video > index)
        if video_idx == 0:
            frame_idx = index
        else:
            frame_idx = index - self.frames_per_video[video_idx - 1]
        label = self.samples['labels'][video_idx]
        label_one_hot_encoded = np.zeros(self.num_classes, dtype=np.float32)
        label_one_hot_encoded[label] = 1
        label_one_hot_encoded = torch.from_numpy(label_one_hot_encoded)

        # get a clip starting at requested frame
        clip = self.get_clip(self.samples['timestamps'][video_idx], frame_idx, self.samples['video_paths'][video_idx])
        # convert to float and dimensions from [T X H X W X C] -> [C X T X H X W]
        clip = clip.float().permute(3, 0, 1, 2) / 255.0
        if self.transform is not None:
            clip = self.transform(clip)
        # convert pixel values to range [-1., 1.]
        clip = (clip * 2.0) - 1.0
        return clip, label_one_hot_encoded, video_idx

    def __len__(self):
        return self.frames_per_video[-1]

    def get_clip(self, frame_timestamps, start_frame, video_path):
        end_frame = start_frame + self.clip_length - 1
        if end_frame >= len(frame_timestamps):
            end_frame = len(frame_timestamps) - 1
        start_pts = int(frame_timestamps[start_frame])
        end_pts = int(frame_timestamps[end_frame])
        (clip,_,_) = tio.read_video(video_path, start_pts = start_pts, end_pts = end_pts)

        # copy last frame so we have a clip with the required length
        padding = self.clip_length - clip.shape[0]
        if padding > 0:
            last_row = clip[-1].unsqueeze(0)
            last_row = last_row.repeat(padding, 1, 1, 1)
            clip = torch.cat((clip, last_row))
        return clip
