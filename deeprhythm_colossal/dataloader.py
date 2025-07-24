#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data loading classes for both full songs and individual clips"""

import random

import numpy as np
import pandas as pd
import torch
from loguru import logger

from deeprhythm_colossal import utils


class ClipDataset(torch.utils.data.Dataset):
    """
    Dataloader for working with clips.

    Parameters:
        split_df (pd.DataFrame): a dataframe containing filepaths, BPM values, etc. for this split
        n_tracks (int, optional): maximum number of tracks to use during training
        clip_length (int, optional): duration of every clip
        sample_rate (int, optional): sample rate to resample clips to
    """

    def __init__(
            self,
            split_df: pd.DataFrame,
            n_clips: int = None,
            clip_length: int = utils.CLIP_LENGTH,
            sample_rate: int = utils.SAMPLE_RATE
    ):
        # Parse sample rate + clip length, then use this to estimate N samples in each clip
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.clip_samples = clip_length * sample_rate

        # TODO: if we're using different buckets/sources, we should also get the "source" of each track from the column
        self.bucket = utils.get_bucket(utils.EXTERNAL_BEATS_BUCKET)

        # Get the filepaths and ground-truth BPMs from the dataframe
        self.df = self.format_df(split_df)

        self.clips = self.get_all_clips()

        # If required, shuffle and get N clips only
        if n_clips is not None:
            random.shuffle(self.clips)
            self.clips = self.clips[:n_clips]

    def format_df(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats and sanitises an input dataframe.

        Removes invalid tracks (i.e., those outside the range of training BPM, or where the header-extracted BPM is a
        mismatch with the one given on beatstars) and adds a few additional metadata columns.
        """

        # Make a copy for safety
        tmp = split_df.copy(deep=True)

        # Drop tracks where the BPM is outside the range
        tmp = tmp[(utils.MIN_BPM <= tmp["bpm"]) & (utils.MAX_BPM >= tmp["bpm"])]

        # Drop tracks where the header BPM does not agree with the beatstars BPM (only when header info is provided)
        bpm_matches_header = tmp["header_bpm"].isna() | (tmp["header_bpm"] == tmp["bpm"])
        mismatch = tmp[~bpm_matches_header]
        logger.warning(f"... found {len(mismatch)} tracks where header contradicts BeatStars BPM, dropping!")
        tmp = tmp[bpm_matches_header]

        # Estimate the framecount for all tracks based on duration and sample rate
        #  We don't care about the extracted sample rate as we'll be resampling anyway
        tmp["framecount_estimated"] = (self.sample_rate * tmp["duration"]).astype(int)

        # Estimate the number of clips we can get from the track based on the frame count
        tmp["clipcount_estimated"] = (tmp["framecount_estimated"] + self.clip_samples - 1) // self.clip_samples
        tmp["clipcount_estimated"] = tmp["clipcount_estimated"].astype(int)

        return tmp

    def get_all_clips(self) -> list[tuple[str, int, int, int]]:
        """
        Gets metadata + slice indices for all clips.

        The return is a list of tuples in the form (fpath, true_bpm, clip_start, clip_end, estimated_samples), where
        `clip_start` and `clip_end` are given in samples with respect to the desired clip length and sample rate.
        """
        allres = []

        # Iterate over every track in the dataset
        for idx, track in self.df.iterrows():

            # Iterate over the predicted number of clips for the track
            for clip_idx in range(track["clipcount_estimated"]):

                # Get the starting point for the clip, in samples
                start = int(self.clip_samples * clip_idx)

                # Break out once we've exceeded the total length of the clip
                if start > track["framecount_estimated"]:
                    break

                # Append everything to the list
                clip_meta = (track["object_fpath"], track["bpm"], start, track["framecount_estimated"])
                allres.append(clip_meta)

        return allres

    def __len__(self) -> int:
        """
        Returns the total number of items in the dataloader.

        This is equivalent to the total number of clips estimated for every track contained within the data subset.
        """
        return self.df["clipcount_estimated"].sum()

    # noinspection PyUnresolvedReferences
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Gets the audio for a clip and BPM as a class index
        """
        # Unpack the tuple
        object_path, true_bpm, clip_start, predicted_framecount = self.clips[idx]

        # Convert the actual BPM to a class IDX for use in training
        class_idx_bpm = utils.bpm_to_class(true_bpm)

        # Make the S3 request to get the audio file
        audiofile = utils.read_s3_object_to_audiofile(object_path, self.bucket)

        # Resample the audio file to the target sample rate
        audiofile_res = audiofile.resampled_to(self.sample_rate)
        total_framecount = audiofile_res.frames

        # Get the number of samples we want to use
        #  If we don't have enough samples left in the audio to have a "complete" clip, just use all the samples left
        if total_framecount - clip_start < self.clip_samples:
            samples_to_read = total_framecount - clip_start
        #  Otherwise, we have a "complete" clip and can use all the samples
        else:
            samples_to_read = self.clip_samples

        # Then, seek to the clip starting point, read the required number of samples, and convert to mono
        audiofile_res.seek(clip_start)
        audiofile_read = audiofile_res.read(samples_to_read).mean(axis=0)

        # If the clip is too short, right-pad to the expected number of samples
        if len(audiofile_read) < self.clip_samples:
            pad_width = self.clip_samples - len(audiofile_read)
            audiofile_read = np.pad(audiofile_read, (0, pad_width), mode='constant', constant_values=0.)

        # Return the audiofile (as a tensor) and the ground truth BPM
        #  We'll do the feature extraction in batches prior to passing to the model
        return torch.from_numpy(audiofile_read), class_idx_bpm


if __name__ == "__main__":
    from time import time

    df = pd.read_csv(utils.get_project_root() / "splits/split_2025-07-24_header/test_split.csv")
    dl = ClipDataset(df, n_clips=1000)

    n_batches = 10
    batch_size = 8
    num_workers = 8
    loader = torch.utils.data.DataLoader(dl, batch_size=batch_size, num_workers=num_workers)

    data_iter = iter(loader)
    batch_times = []
    for batch_idx in range(len(loader)):
        start_time = time()
        batch = next(data_iter)
        end_time = time()
        logger.info(f"Batch {batch_idx + 1}: Took {end_time - start_time:.4f} seconds to prepare")
        batch_times.append(end_time - start_time)
        if batch_idx > n_batches:
            break

    logger.info(f"Average time for {n_batches} batches of size: {np.mean(batch_times):.4f} seconds")
