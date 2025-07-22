#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchaudio
import torchvision
import os
import subprocess


class AVSRDataLoader:
    def __init__(self, modality, detector="retinaface", convert_gray=True, 
                 crop_type="lips", crop_width=96, crop_height=96):
        self.modality = modality
        self.crop_type = crop_type
        
        if modality == "video":
            if detector == "retinaface":
                from detectors.retinaface.detector import LandmarksDetector
                from detectors.retinaface.video_process_enhanced import VideoProcessEnhanced

                self.landmarks_detector = LandmarksDetector(device="cuda:0")
                self.video_process = VideoProcessEnhanced(
                    convert_gray=convert_gray,
                    crop_type=crop_type,
                    crop_width=crop_width,
                    crop_height=crop_height
                )

            if detector == "mediapipe":
                from detectors.mediapipe.detector import LandmarksDetector
                from detectors.mediapipe.video_process_enhanced import VideoProcessEnhanced

                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcessEnhanced(
                    convert_gray=convert_gray,
                    crop_type=crop_type,
                    crop_width=crop_width,
                    crop_height=crop_height
                )

    def load_data(self, data_filename, landmarks=None, transform=True):
        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            return audio
        if self.modality == "video":
            video = self.load_video(data_filename)
            if not landmarks:
                landmarks = self.landmarks_detector(video)
            video = self.video_process(video, landmarks)
            if video is None:
                raise TypeError("video cannot be None")
            video = torch.tensor(video)
            return video

    def load_audio(self, data_filename):
        if data_filename.endswith('.mp4'):
            # Convert mp4 to wav first
            wav_path = data_filename.replace('.mp4', '.wav')
            if not os.path.exists(wav_path):
                try:
                    subprocess.run([
                        'ffmpeg', '-i', data_filename, '-vn', '-acodec', 'pcm_s16le', 
                        '-ar', '16000', '-ac', '1', wav_path, '-y'
                    ], check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    return None, None
            data_filename = wav_path
        
        audio, sample_rate = torchaudio.load(data_filename)
        return audio, sample_rate

    def load_video(self, data_filename):
        video, _, _ = torchvision.io.read_video(data_filename, pts_unit="sec")
        video = video.permute(0, 3, 1, 2).float() / 255.0
        return video

    def audio_process(self, audio, sample_rate):
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio = resampler(audio)
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio
