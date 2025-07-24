#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Inference modules and functions"""

import argparse
import json
import os
import tempfile
import warnings

import torch
import torch.multiprocessing as multiprocessing

from deeprhythm_colossal import utils
from deeprhythm_colossal.audio_proc.hcqm import make_kernels, compute_hcqm
from deeprhythm_colossal.model.frame_cnn import DeepRhythmModel


NUM_WORKERS = 8
NUM_BATCH = 128


class DeepRhythmPredictor:
    def __init__(self, model_path='deeprhythm_colossal-0.5.pth', device=utils.DEVICE, quiet=False):
        self.model_path = utils.get_weights(quiet=quiet)
        self.device = torch.device(device)
        self.model = self.load_model()
        self.specs = self.make_kernels()

    def load_model(self):
        model = DeepRhythmModel()
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(device=self.device)
        model.eval()
        return model

    def make_kernels(self, device=None):
        if device is None:
            device = self.device
        stft, band, cqt = make_kernels(device=device)
        return stft, band, cqt

    def predict(self, filename, include_confidence=False):
        clips = utils.load_and_split_audio(filename, sr=22050)
        input_batch = compute_hcqm(clips.to(device=self.device), *self.specs).permute(0, 3, 1, 2)
        self.model.eval()
        with torch.no_grad():
            input_batch = input_batch.to(device=self.device)
            outputs = self.model(input_batch)
            probabilities = torch.softmax(outputs, dim=1)
            mean_probabilities = probabilities.mean(dim=0)
            confidence_score, predicted_class = torch.max(mean_probabilities, 0)
            predicted_global_bpm = utils.class_to_bpm(predicted_class.item())
        if include_confidence:
            return predicted_global_bpm, confidence_score.item(),
        return predicted_global_bpm

    def predict_from_audio(self, audio, sr, include_confidence=False):
        clips = utils.split_audio(audio, sr)
        input_batch = compute_hcqm(clips.to(device=self.device), *self.specs).permute(0, 3, 1, 2)
        self.model.eval()
        with torch.no_grad():
            input_batch = input_batch.to(device=self.device)
            outputs = self.model(input_batch)
            probabilities = torch.softmax(outputs, dim=1)
            mean_probabilities = probabilities.mean(dim=0)
            confidence_score, predicted_class = torch.max(mean_probabilities, 0)
            predicted_global_bpm = utils.class_to_bpm(predicted_class.item())
        if include_confidence:
            return predicted_global_bpm, confidence_score.item(),
        return predicted_global_bpm

    def predict_batch(self, dirname, include_confidence=False, workers=8, batch=128, quiet=True):
        """
        Predict BPM for all audio files in a directory using efficient batch processing.

        Args:
            dirname: Directory containing audio files
            include_confidence: Whether to include confidence scores in results

        Returns:
            dict: Mapping of filenames to their predicted BPMs (and optionally confidence scores)
        """
        # Create a temporary file to store batch results
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Run batch inference
            main(
                dataset=get_audio_files(dirname),
                data_path=temp_path,
                device=str(self.device),
                conf=include_confidence,
                quiet=quiet,
                n_workers=workers,
                max_len_batch=batch
            )

            # Read and parse results
            results = {}
            with open(temp_path, 'r') as f:
                for line in f:
                    result = json.loads(line.strip())
                    filename = result.pop('filename')
                    if include_confidence:
                        results[filename] = (result['bpm'], result['confidence'])
                    else:
                        results[filename] = result['bpm']

            return results

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def predict_per_frame(self, filename, include_confidence=False):
        clips = utils.load_and_split_audio(filename, sr=22050)
        input_batch = compute_hcqm(clips.to(device=self.device), *self.specs).permute(0, 3, 1, 2)
        self.model.eval()
        with torch.no_grad():
            input_batch = input_batch.to(device=self.device)
            outputs = self.model(input_batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
            predicted_bpms = [utils.class_to_bpm(cls.item()) for cls in predicted_classes]

        if include_confidence:
            return predicted_bpms, confidence_scores.tolist()
        return predicted_bpms


def producer(task_queue, result_queue, completion_event, queue_condition, queue_threshold=NUM_BATCH*2):
    """
    Loads audio, splits it into a list of 8s clips, and puts the clips into the result queue.
    """
    while True:
        task = task_queue.get()
        if task is None:
            result_queue.put(None)  # Send termination signal to indicate this producer is done
            completion_event.wait()  # Wait for the signal to exit
            break
        filename = task
        with queue_condition:  # Use the condition to wait if the queue is too full before loading audio
            while result_queue.qsize() >= queue_threshold:
                queue_condition.wait()
        clips = utils.load_and_split_audio(filename, share_mem=True)
        if clips is not None:
            result_queue.put((clips, filename))

def init_workers(dataset, n_workers=NUM_WORKERS):
    """
    Initializes worker processes for multiprocessing, setting up the required queues,
    an event for coordinated exit, and a condition for queue threshold management.

    Parameters:
    - n_workers: Number of worker processes to start.
    - dataset: The dataset items to process.
    - queue_threshold: The threshold for the result queue before producers wait.
    """
    manager = multiprocessing.Manager()
    task_queue = multiprocessing.Queue()
    result_queue = manager.Queue()  # Managed Queue for sharing across processes
    completion_event = manager.Event()
    queue_condition = manager.Condition()
    producers = [
        multiprocessing.Process(
            target=producer,
            args=(task_queue, result_queue, completion_event, queue_condition)
        ) for _ in range(n_workers)
    ]
    for p in producers:
        p.start()
    for item in dataset:
            task_queue.put(item)
    for _ in range(n_workers):
        task_queue.put(None)

    return task_queue, result_queue, producers, completion_event, queue_condition


def process_and_save(batch_audio, batch_meta, specs, model, out_path, conf=False, quiet=False):
    """
    Processes a batch of audio clips and saves the result along with metadata to an HDF5 file.
    """
    stft, band, cqt = specs
    hcqm = compute_hcqm(batch_audio, stft, band, cqt)
    model_device = next(model.parameters()).device
    if not quiet:
        print('hcqm done', hcqm.shape)
    with torch.no_grad():
        hcqm = hcqm.permute(0,3,1,2).to(device=model_device)
        outputs = model(hcqm)
        if not quiet:
            print('model done', outputs.shape)
    torch.cuda.empty_cache()
    results = []
    for meta in batch_meta:
        filename, num_clips, start_idx = meta
        song_outputs = outputs[start_idx:start_idx+num_clips, :]
        probabilities = torch.softmax(song_outputs, dim=1)
        mean_probabilities = probabilities.mean(dim=0)
        confidence_score, predicted_class = torch.max(mean_probabilities, 0)
        predicted_global_bpm = utils.class_to_bpm(predicted_class.item())
        result = {
            "filename": filename,
            "bpm": predicted_global_bpm
        }
        if conf:
            result['confidence'] = confidence_score.item()
        results.append(result)
    with open(out_path, 'a') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def consume_and_process(result_queue, data_path, queue_condition, n_workers=NUM_WORKERS, max_len_batch=NUM_BATCH, device='cuda', conf=False, quiet=False):
    batch_audio = []
    batch_meta = []
    active_producers = n_workers
    sr = 22050
    len_audio = sr * 8
    if not quiet:
        print(f'Using device: {device}')
    specs = make_kernels(len_audio,  sr, device=device)
    if not quiet:
        print('made kernels')
    model = load_cnn_model(device=device, quiet=quiet)
    model.eval()
    if not quiet:
        print('loaded model')
    total_clips = 0
    if not quiet:
        print(f'producers = {active_producers}')
    while active_producers > 0:
        result = result_queue.get()
        with queue_condition:
            queue_condition.notify_all()
        if result is None:
            active_producers -= 1
            if not quiet:
                print(f'producers = {active_producers}')
            continue
        clips, filename = result
        batch_audio.append(clips)
        num_clips = clips.shape[0]
        start_idx = total_clips
        batch_meta.append((filename, num_clips, start_idx))
        total_clips += num_clips
        if total_clips >= max_len_batch:
            stacked_batch_audio = torch.cat(batch_audio, dim=0).to(device=device)
            process_and_save(stacked_batch_audio, batch_meta, specs,model, data_path, conf=conf, quiet=quiet)
            total_clips = 0
            batch_audio = []
            batch_meta = []

    # Make sure to process any remaining clips
    if batch_audio:
        stacked_batch_audio = torch.cat(batch_audio, dim=0).to(device=device)
        process_and_save(stacked_batch_audio, batch_meta, specs,model, data_path, conf=conf, quiet=quiet)
        pass


def load_cnn_model(path='deeprhythm_colossal-0.7.pth', device=None, quiet=False):
    model = DeepRhythmModel(256)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(path):
        path = utils.get_weights(quiet=quiet)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model = model.to(device=device)
    model.eval()
    return model


def get_audio_files(dir_path):
    """
    Collects all audio files recursively from a specified directory.
    """
    audio_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))
    return audio_files


def main(dataset, n_workers=NUM_WORKERS, max_len_batch=NUM_BATCH, data_path='output.jsonl', device='cuda', conf=False, quiet=False):
    task_queue, result_queue, producers, completion_event, queue_condition = init_workers(dataset, n_workers)
    try:
        consume_and_process(result_queue, data_path, queue_condition, n_workers=n_workers,max_len_batch=max_len_batch, device=device, conf=conf, quiet=quiet)
    finally:
        completion_event.set()
        for p in producers:
            p.join()  # Ensure all producer processes have finished


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Path to the audio file to analyze')
    parser.add_argument('-d','--device', type=str, default=utils.DEVICE, help='Device to use for inference')
    parser.add_argument('-c','--conf', action='store_true', help='Include confidence score in output')
    parser.add_argument('-q','--quiet', action='store_true', help='Use minimal output format')
    args = parser.parse_args()


    predictor = DeepRhythmPredictor(device=args.device, quiet=args.quiet)
    result = predictor.predict(args.filename, include_confidence=args.conf)
    if args.conf:
        bpm, conf = result
    else:
        bpm = result
    
    if args.quiet:
        print(result)
    else:
        print(f'Predicted BPM: {bpm}')
        if args.conf:
            print(f'Confidence: {conf}')
