#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os

import cv2
import numpy as np
from skimage import transform as tf


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks


def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform("similarity", src, dst)
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    warped = (warped * 255).astype("uint8")
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = (warped * 255).astype("uint8")
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)
    # Check for too much bias in height and width
    if abs(center_y - img.shape[0] / 2) > height + threshold:
        raise OverflowError("too much bias in height")
    if abs(center_x - img.shape[1] / 2) > width + threshold:
        raise OverflowError("too much bias in width")
    # Calculate bounding box coordinates
    y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
    # Cut the image
    cutted_img = np.copy(img[y_min:y_max, x_min:x_max])
    return cutted_img


def cut_face_patch(img, landmarks, height, width, threshold=20):
    """Cut face region using all facial landmarks"""
    # Use all face landmarks to determine face boundary
    face_landmarks = landmarks[:17]  # Face contour landmarks
    if len(face_landmarks) == 0:
        # Fallback to all landmarks if face contour not available
        face_landmarks = landmarks
    
    # Calculate face center and boundaries
    center_x, center_y = np.mean(face_landmarks, axis=0)
    
    # Calculate face dimensions
    face_width = np.max(face_landmarks[:, 0]) - np.min(face_landmarks[:, 0])
    face_height = np.max(face_landmarks[:, 1]) - np.min(face_landmarks[:, 1])
    
    # Use larger crop to include full face
    crop_width = max(width, int(face_width * 1.5))
    crop_height = max(height, int(face_height * 1.5))
    
    # Calculate bounding box coordinates
    y_min = int(round(np.clip(center_y - crop_height//2, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + crop_height//2, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - crop_width//2, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + crop_width//2, 0, img.shape[1])))
    
    # Cut the image
    cutted_img = np.copy(img[y_min:y_max, x_min:x_max])
    
    # Resize to target size
    cutted_img = cv2.resize(cutted_img, (width, height))
    return cutted_img


class VideoProcessEnhanced:
    def __init__(
        self,
        mean_face_path="20words_mean_face.npy",
        crop_width=96,
        crop_height=96,
        start_idx=3,
        stop_idx=4,
        window_margin=12,
        convert_gray=True,
        crop_type="lips"
    ):
        self.reference = np.load(
            os.path.join(os.path.dirname(__file__), mean_face_path)
        )
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray
        self.crop_type = crop_type
        
        # Adjust landmark indices based on crop type
        if crop_type == "lips":
            self.start_idx = 48  # Mouth landmarks start
            self.stop_idx = 68   # Mouth landmarks end
        elif crop_type == "face":
            self.start_idx = 0   # All landmarks
            self.stop_idx = 68   # All landmarks
        # For "full", we won't use landmarks for cropping

    def __call__(self, video, landmarks):
        if self.crop_type == "full":
            # Return full video frames without cropping
            return self.process_full_video(video)
        
        # Pre-process landmarks: interpolate frames that are not detected
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        # Exclude corner cases: no landmark in all frames
        if not preprocessed_landmarks:
            return
        # Affine transformation and crop patch
        sequence = self.crop_patch(video, preprocessed_landmarks)
        assert sequence is not None, "crop an empty patch."
        return sequence

    def process_full_video(self, video):
        """Process video without cropping - return full frames"""
        sequence = []
        for frame in video:
            # Convert to numpy if needed
            if hasattr(frame, 'numpy'):
                frame = frame.numpy()
            elif hasattr(frame, 'cpu'):
                frame = frame.cpu().numpy()
            
            # Ensure correct shape (H, W, C)
            if frame.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                frame = np.transpose(frame, (1, 2, 0))
            
            # Convert to uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Convert to grayscale if needed
            if self.convert_gray and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            sequence.append(frame)
        return np.array(sequence)

    def crop_patch(self, video, landmarks):
        sequence = []
        for frame_idx, frame in enumerate(video):
            # Convert tensor to numpy if needed
            if hasattr(frame, 'numpy'):
                frame = frame.numpy()
            elif hasattr(frame, 'cpu'):
                frame = frame.cpu().numpy()
            
            # Ensure correct shape (H, W, C)
            if frame.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                frame = np.transpose(frame, (1, 2, 0))
            
            # Convert to uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            window_margin = min(
                self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx
            )
            smoothed_landmarks = np.mean(
                [
                    landmarks[x]
                    for x in range(
                        frame_idx - window_margin, frame_idx + window_margin + 1
                    )
                ],
                axis=0,
            )
            smoothed_landmarks += landmarks[frame_idx].mean(
                axis=0
            ) - smoothed_landmarks.mean(axis=0)
            
            transformed_frame, transformed_landmarks = self.affine_transform(
                frame, smoothed_landmarks, self.reference, grayscale=self.convert_gray
            )
            
            if self.crop_type == "lips":
                patch = cut_patch(
                    transformed_frame,
                    transformed_landmarks[self.start_idx : self.stop_idx],
                    self.crop_height // 2,
                    self.crop_width // 2,
                )
            elif self.crop_type == "face":
                patch = cut_face_patch(
                    transformed_frame,
                    transformed_landmarks,
                    self.crop_height,
                    self.crop_width,
                )
            
            sequence.append(patch)
        return np.array(sequence)

    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(
                    landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
                )

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        # Handle corner case: keep frames at the beginning or at the end that failed to be detected
        if valid_frames_idx:
            landmarks[: valid_frames_idx[0]] = [
                landmarks[valid_frames_idx[0]]
            ] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
                len(landmarks) - valid_frames_idx[-1]
            )

        assert all(lm is not None for lm in landmarks), "not every frame has landmark"

        return landmarks

    def affine_transform(
        self,
        frame,
        landmarks,
        reference,
        grayscale=False,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(33, 36, 39, 42, 45),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
    ):
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stable_reference = self.get_stable_reference(
            reference, reference_size, target_size
        )
        transform = self.estimate_affine_transform(
            landmarks, stable_points, stable_reference
        )
        transformed_frame, transformed_landmarks = self.apply_affine_transform(
            frame,
            landmarks,
            transform,
            target_size,
            interpolation,
            border_mode,
            border_value,
        )

        return transformed_frame, transformed_landmarks

    def get_stable_reference(self, reference, reference_size, target_size):
        # -- right eye, left eye, nose tip, mouth center
        stable_reference = np.vstack(
            [
                np.mean(reference[36:42], axis=0),
                np.mean(reference[42:48], axis=0),
                np.mean(reference[31:36], axis=0),
                np.mean(reference[48:68], axis=0),
            ]
        )
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference

    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        return cv2.estimateAffinePartial2D(
            np.vstack([landmarks[x] for x in stable_points]),
            stable_reference,
            method=cv2.LMEDS,
        )[0]

    def apply_affine_transform(
        self,
        frame,
        landmarks,
        transform,
        target_size,
        interpolation,
        border_mode,
        border_value,
    ):
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = (
            np.matmul(landmarks, transform[:, :2].transpose())
            + transform[:, 2].transpose()
        )
        return transformed_frame, transformed_landmarks
