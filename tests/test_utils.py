#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests AWS configuration"""

import os

import numpy as np
import pandas as pd
import pytest
from pedalboard.io import AudioFile

from deeprhythm_colossal import utils


TEST_FILE = os.path.join(utils.get_project_root(), "data/1000028.mp3")


@pytest.mark.parametrize(
    "bucket_name",
    [
        "cl-dev-reference-tracks",
        "cl-stage-cdn-logs",
        "cl-stage-cdn-origin",
        "cl-stage-model-storage",
        utils.EXTERNAL_BEATS_BUCKET
    ]
)
def test_get_bucket(bucket_name):
    bucket_ = utils.get_bucket(bucket_name)
    assert hasattr(bucket_, 'name') and bucket_.__class__.__name__ == 's3.Bucket'


@pytest.mark.parametrize(
    "object_name",
    [
        "11047732.mp3",
    ]
)
def test_get_object(object_name):
    # Get a bucket we know to exist
    bucket = utils.get_bucket(utils.EXTERNAL_BEATS_BUCKET)
    object_ = utils.get_object(object_name, bucket)
    assert isinstance(object_, dict)
    assert "Body" in object_
    assert object_["ResponseMetadata"]["HTTPStatusCode"] == 200


@pytest.mark.parametrize(
    "obj,bucket",
    [
        ("11047732.mp3", utils.EXTERNAL_BEATS_BUCKET),
    ]
)
def test_get_audiofile(obj, bucket):
    af = utils.read_s3_object_to_audiofile(obj, bucket)
    assert isinstance(af, AudioFile)
    assert hasattr(af, "resampled_to")


@pytest.mark.parametrize(
    "table_name,columns",
    [
        ("external_beat", ["id", "artist", "title"])
    ]
)
def test_read_database(table_name, columns):
    engine = utils.get_db_engine()
    df = pd.read_sql_table(table_name, engine, columns=columns)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == columns


@pytest.mark.parametrize(
    "bpm, expected",
    [
        (50, 0),   # first class in array
        (55, 5),
        (100, 50),    # middle class in array
        (200, 150),     # last class in array
        (196, 146),
        (10, IndexError),
        (0, IndexError),
        (9000, IndexError)
    ]
)
def test_bpm_to_class(bpm, expected):
    if expected is IndexError:
        with pytest.raises(expected, match=f"BPM value {bpm} not found in bpm_classes."):
            _ = utils.bpm_to_class(bpm)
    else:
        assert utils.bpm_to_class(bpm, bpm_classes=utils.BPMS) == expected


@pytest.mark.parametrize(
    "class_index, expected",
    [
        (0, 50),           # first BPM
        (5, 55),
        (50, 100),         # middle BPM
        (150, 200),        # last BPM
        (146, 196),
        (-1, IndexError),  # invalid: negative index
        (151, IndexError), # invalid: out of bounds (just above max)
        (200, IndexError), # invalid: well out of bounds
    ]
)
def test_class_to_bpm(class_index, expected):
    if expected is IndexError:
        with pytest.raises(IndexError):
            utils.class_to_bpm(class_index, bpm_classes=utils.BPMS)
    else:
        assert utils.class_to_bpm(class_index, bpm_classes=utils.BPMS) == expected


@pytest.mark.parametrize(
    "input_lists, expected",
    [
        ((["a", "b"], ["c", "d"]), True),              # completely unique
        ((["a", "b"], ["b", "c"]), False),             # overlap on "b"
        (([], ["x"]), True),                            # empty and non-empty unique
        ((["x"], ["x"]), False),                        # identical single-element lists
        ((["a", "b"], ["c", "d"], ["e", "f"]), True), # multiple unique lists
        ((["a"], ["b"], ["a"]), False),                 # overlap in first and third
        (([],), True),                                  # single empty list
        ((), True),                                     # no lists at all
    ]
)
def test_lists_are_unique(input_lists, expected):
    assert utils.lists_are_unique(*input_lists) is expected


@pytest.mark.parametrize(
    "audio_seconds,expected",
    [
        (0.0, ValueError),  # empty audio, expect 0 clips
        (0.5, 1),  # shorter than one clip -> one padded clip
        (2.0, 1),
        (8.0, 1),
        (8.1, 2),
        (15.9, 2),  # nearly 2 clips
        (16.0, 2),
        (59.4, 8),
        (64.0, 8),  # exactly 8 clips of 8s each
        (65.0, 9),  # one clip more, last padded
    ]
)
def test_split_audio(audio_seconds, expected):
    mono = np.random.random(int(audio_seconds * utils.SAMPLE_RATE)).astype(np.float32)
    if expected is ValueError:
        with pytest.raises(expected):
            _ = utils.split_audio(mono)
    else:
        splitted = utils.split_audio(mono)
        assert splitted.ndim == 2
        actual_clips, actual_samples = splitted.shape
        expected_clips = expected
        assert actual_clips == expected_clips
        assert actual_samples == utils.SAMPLE_RATE * utils.CLIP_LENGTH


@pytest.mark.parametrize(
    "audio_fpath,expected",
    [
        # All of these audios have 30 seconds, so that leads to 4 clips of 8 seconds (with the last containing padding)
        (utils.get_project_root() / "tests/test_resources/audio/000010.mp3", 4),
        (utils.get_project_root() / "tests/test_resources/audio/001666.mp3", 4),
        (utils.get_project_root() / "tests/test_resources/audio/007527.mp3", 4)
    ]
)
def test_load_and_split_audio(audio_fpath, expected):
    splitted = utils.load_and_split_audio(audio_fpath)
    actual_clips, actual_samples = splitted.shape
    expected_clips = expected
    assert actual_clips == expected_clips
    assert actual_samples == utils.SAMPLE_RATE * utils.CLIP_LENGTH
    # Check that the final few samples of the last clip are padded
    should_be_zeroes = splitted[-1, -utils.SAMPLE_RATE:]
    assert not should_be_zeroes.any()
