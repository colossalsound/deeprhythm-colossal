#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data loading classes for both full songs and individual clips"""

import h5py
import torch
from torch.utils.data import Dataset

from deeprhythm_colossal.utils import bpm_to_class


def song_collate(batch):
    # Each element in `batch` is a tuple (song_clips, global_bpm)
    # Where song_clips is a tensor of shape [num_clips, 240, 8, 6]
    inputs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return inputs, labels


class ClipDataset(Dataset):
    def __init__(self, hdf5_file, group, use_float=False):
        """
        :param hdf5_file: Path to the HDF5 file.
        :param group: Group in the HDF5 file to use ('train', 'test', 'validate').
        """
        self.use_float = use_float
        self.hdf5_file = hdf5_file
        self.group = group
        self.index_map = []
        self.file_ref = h5py.File(self.hdf5_file, 'r')
        group_data = self.file_ref[group]
        for song_key in group_data.keys():
            song_data = group_data[song_key]
            if song_data.attrs['source'] == 'fma':
                continue
            num_clips = song_data['hcqm'].shape[0]
            if num_clips > 5:
                clip_start = 1
                clip_range = num_clips-2
            else:
                clip_start, clip_range = 0, num_clips

            for clip_index in range(clip_start, clip_range):
                self.index_map.append((song_key, clip_index))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        song_key, clip_index = self.index_map[idx]
        song_data = self.file_ref[self.group][song_key]
        hcqm = song_data['hcqm'][clip_index]
        bpm = torch.tensor(float(song_data.attrs['bpm']), dtype=torch.float32)
        hcqm_tensor = torch.tensor(hcqm, dtype=torch.float).permute(2, 0, 1)
        if self.use_float:
            return hcqm_tensor, bpm
        label_class_index = bpm_to_class(int(bpm))  # Convert BPM to class index
        return hcqm_tensor, label_class_index


class SongDataset(Dataset):
    def __init__(self, hdf5_path, group):
        """
        Args:
            hdf5_path (str): Path to the HDF5 file.
            group (str): Group in HDF5 file ('train', 'test', 'validate').
        """
        super(SongDataset, self).__init__()
        self.hdf5_path = hdf5_path
        self.group = group
        self.file = h5py.File(hdf5_path, 'r')
        self.group_file = self.file[group]
        self.keys = []
        for key in self.group_file.keys():
            if self.group_file[key].attrs['source'] == 'fma':
                continue
            else:
                self.keys.append(key)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        song_key = self.keys[idx]
        song_data = self.group_file[song_key]
        hcqm = torch.tensor(song_data['hcqm'][:])
        bpm_class = bpm_to_class(int(float(song_data.attrs['bpm'])))
        return hcqm, bpm_class

    def close(self):
        self.file.close()
