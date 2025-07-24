#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests AWS configuration"""

import os

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
        "cl-dev-external-beats"
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
    bucket = utils.get_bucket("cl-dev-external-beats")
    object_ = utils.get_object(object_name, bucket)
    assert hasattr(object_, 'key') and object_.__class__.__name__ == 's3.Object'
    # Try getting the object, should get a OK status code
    getted = object_.get()
    assert getted["ResponseMetadata"]["HTTPStatusCode"] == 200


@pytest.mark.parametrize(
    "obj,bucket",
    [
        ("11047732.mp3", "cl-dev-external-beats"),
        (utils.get_object("11047732.mp3", "cl-dev-external-beats"), "cl-dev-external-beats"),
        ("11047732.mp3", utils.get_bucket("cl-dev-external-beats"))
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
