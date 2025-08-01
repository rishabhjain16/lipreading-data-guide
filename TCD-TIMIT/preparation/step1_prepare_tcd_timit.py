#!/usr/bin/env python3
"""
TCD-TIMIT Dataset Preprocessing Pipeline (HD-Optimized)
=======================================================

An optimized preprocessing script for TCD-TIMIT dataset that:
1. Leverages HD quality (1920x1080) for superior face detection
2. Outputs training-friendly resolutions (96x96 lips, 224x224 face)
3. Balances quality with computational efficiency

Key Features:
- HD-quality face detection for precise lip region extraction
- Configurable output resolutions optimized for training
- Efficient processing pipeline
- Compatible with existing LRS2/LRS3 model architectures

Usage:
    python step1_prepare_tcd_timit_optimized.py \
        --data-dir /path/to/TCD-TIMIT \
        --root-dir /path/to/output \
        --subset volunteers \
        --crop-type lips \
        --output-size 96
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
import subprocess
warnings.filterwarnings("ignore")

# Import data module for detector support
from data.data_module import AVSRDataLoader

# TCD-TIMIT sentence mapping (TIMIT corpus standard sentences)
import json

with open("./timit_sentences.json", "r", encoding="utf-8") as f:
    TIMIT_SENTENCES = json.load(f)

def parse_mlf_transcripts(mlf_file_path):
    """Parse .mlf file to extract transcript mappings"""
    
    transcripts = {}
    current_file = None
    current_phonemes = []
    
    with open(mlf_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line == "#!MLF!#":
                continue
                
            if line.startswith('"') and line.endswith('.rec"'):
                # New file entry - save previous if exists
                if current_file and current_phonemes:
                    transcript = ' '.join(current_phonemes)
                    transcripts[current_file] = transcript
                
                # Extract transcript ID from filename
                current_file = line[1:-5]  # Remove quotes and .rec extension
                current_file = os.path.basename(current_file)  # Get just the filename
                current_phonemes = []
                
            elif line == ".":
                # End of current file's transcript
                if current_file and current_phonemes:
                    transcript = ' '.join(current_phonemes)
                    transcripts[current_file] = transcript
                current_file = None
                current_phonemes = []
                
            else:
                # Phoneme entry: start_time end_time phoneme
                parts = line.split()
                if len(parts) == 3:
                    start_time, end_time, phoneme = parts
                    # Skip silence markers
                    if phoneme.lower() not in ['sil', 'sp']:
                        current_phonemes.append(phoneme)
    
    # Handle last entry if file doesn't end with '.'
    if current_file and current_phonemes:
        transcript = ' '.join(current_phonemes)
        transcripts[current_file] = transcript
    
    return transcripts

def get_landmarks_detector(detector_type):
    """
    Get the appropriate landmarks detector based on the specified type.
    Returns a detector object that can process video frames.
    """
    if detector_type == "mediapipe":
        from detectors.mediapipe.detector import LandmarksDetector
        return LandmarksDetector()
    elif detector_type == "retinaface":
        from detectors.retinaface.detector import LandmarksDetector
        return LandmarksDetector(device="cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # "opencv" - fallback to original implementation
        return OpenCVFaceDetector()

class OpenCVFaceDetector:
    """
    OpenCV-based face detector (fallback) with a compatible interface
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def __call__(self, video_frames):
        landmarks = []
        for frame in video_frames:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with improved parameters
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Create simple landmarks at corners of mouth area (approximation)
                # These aren't real facial landmarks but provide a compatible interface
                mouth_y = y + int(h * 0.7)  # Approximate mouth position
                landmarks.append([
                    [x + int(w * 0.3), mouth_y - int(h * 0.1)],  # Left corner
                    [x + int(w * 0.7), mouth_y - int(h * 0.1)],  # Right corner
                    [x + int(w * 0.5), mouth_y + int(h * 0.1)],  # Bottom center
                    [x + int(w * 0.5), mouth_y - int(h * 0.1)]   # Top center
                ])
            else:
                landmarks.append(None)
                
        return landmarks

def smart_crop_hd_video(frame, crop_type="lips", output_size=96, detector_instance=None):
    """
    Intelligently crop HD video using face detection
    Takes advantage of 1920x1080 resolution for better detection
    Outputs training-friendly resolution
    """
    if detector_instance is None:
        # Fallback to OpenCV detector
        detector_instance = OpenCVFaceDetector()
        landmarks = detector_instance([frame])[0]
    else:
        # Use provided detector instance
        landmarks = detector_instance([frame])[0]
    
    h, w = frame.shape[:2]
    
    if landmarks is not None:
        if crop_type == "lips":
            # Extract lip region from detected landmarks
            try:
                # Different processing based on detector type (they provide different landmarks)
                if isinstance(detector_instance, OpenCVFaceDetector):
                    # Simple 4-point mouth landmarks from OpenCV
                    center_x = np.mean([p[0] for p in landmarks])
                    center_y = np.mean([p[1] for p in landmarks])
                    
                    # Calculate tight crop around lips
                    lip_height = int(output_size * 1.2)
                    lip_width = int(output_size * 1.2)
                    
                elif hasattr(detector_instance, 'short_range_detector'):  # MediaPipe detector
                    # MediaPipe provides 4 points - the 4th one is the mouth
                    mouth_x, mouth_y = landmarks[3]
                    
                    # Create a tighter crop around the mouth point
                    center_x = mouth_x
                    center_y = mouth_y
                    
                    # Make the lip crop proportional to the face size
                    face_width = max([p[0] for p in landmarks]) - min([p[0] for p in landmarks])
                    lip_width = int(face_width * 0.5)  # Half the face width
                    lip_height = int(lip_width * 0.8)  # Slightly narrower height
                    
                else:  # RetinaFace detector with 68 landmarks
                    # Get only the mouth landmarks (indices 48-68)
                    mouth_landmarks = landmarks[48:68]
                    
                    # Calculate the center and size of the mouth region
                    mouth_x = np.mean([p[0] for p in mouth_landmarks])
                    mouth_y = np.mean([p[1] for p in mouth_landmarks])
                    
                    # Get the width and height of the mouth
                    mouth_width = max([p[0] for p in mouth_landmarks]) - min([p[0] for p in mouth_landmarks])
                    mouth_height = max([p[1] for p in mouth_landmarks]) - min([p[1] for p in mouth_landmarks])
                    
                    # Center point
                    center_x = mouth_x
                    center_y = mouth_y
                    
                    # Add some margin around the mouth
                    lip_width = int(mouth_width * 1.5)
                    lip_height = int(mouth_height * 2.0)  # More vertical margin
                
                # Apply the crop centered on the lips
                lip_x = int(center_x - lip_width // 2)
                lip_y = int(center_y - lip_height // 2)
                
                # Ensure coordinates are within frame
                lip_x = max(0, min(lip_x, w - 1))
                lip_y = max(0, min(lip_y, h - 1))
                lip_width = min(lip_width, w - lip_x)
                lip_height = min(lip_height, h - lip_y)
                
                if lip_width > 0 and lip_height > 0:
                    cropped = frame[lip_y:lip_y+lip_height, lip_x:lip_x+lip_width]
                else:
                    # Fallback to center crop
                    cropped = center_crop_fallback(frame, crop_type, output_size)
            except Exception as e:
                print(f"Landmark processing failed: {e}")
                # Fallback if landmarks processing fails
                cropped = center_crop_fallback(frame, crop_type, output_size)
                
        elif crop_type == "face":
            # Different face crop method based on detector type
            if isinstance(detector_instance, OpenCVFaceDetector) or hasattr(detector_instance, 'short_range_detector'):
                # For OpenCV and MediaPipe detectors
                # Calculate bounding box around all landmarks
                x_min = min([p[0] for p in landmarks])
                y_min = min([p[1] for p in landmarks])
                x_max = max([p[0] for p in landmarks])
                y_max = max([p[1] for p in landmarks])
                
                width = x_max - x_min
                height = y_max - y_min
                
                # Add padding
                padding = int(max(width, height) * 0.4)  # More padding for face
                face_x = max(0, x_min - padding)
                face_y = max(0, y_min - padding)
                face_w = min(w - face_x, width + 2*padding)
                face_h = min(h - face_y, height + 2*padding)
            else:
                # For RetinaFace with 68 landmarks
                # Get the face contour landmarks
                face_contour = landmarks[:17]  # Jaw landmarks
                
                # Include eyes and eyebrows
                left_eyebrow = landmarks[17:22]
                right_eyebrow = landmarks[22:27]
                
                # Calculate bounding box
                face_points = np.vstack([face_contour, left_eyebrow, right_eyebrow])
                x_min = np.min(face_points[:, 0])
                y_min = np.min(face_points[:, 1])
                x_max = np.max(face_points[:, 0])
                y_max = np.max(face_points[:, 1])
                
                width = x_max - x_min
                height = y_max - y_min
                
                # Add some padding
                padding_w = int(width * 0.2)
                padding_h = int(height * 0.2)
                face_x = max(0, x_min - padding_w)
                face_y = max(0, y_min - padding_h)
                face_w = min(w - face_x, width + 2*padding_w)
                face_h = min(h - face_y, height + 2*padding_h)
            
            cropped = frame[face_y:face_y+face_h, face_x:face_x+face_w]
            
        else:  # full
            cropped = frame
    else:
        # Fallback to center crop if no landmarks detected
        cropped = center_crop_fallback(frame, crop_type, output_size)
    
    # Resize to target output size
    if cropped.size > 0:
        target_size = (output_size, output_size)
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
        return resized
    else:
        # Final fallback
        return center_crop_fallback(frame, crop_type, output_size)

def center_crop_fallback(frame, crop_type="lips", output_size=96):
    """Fallback center crop when face detection fails"""
    h, w = frame.shape[:2]
    
    if crop_type == "lips":
        # Crop bottom-center for lips
        center_x, center_y = w // 2, int(h * 0.75)  # Lower center
    elif crop_type == "face":
        # Crop center for face
        center_x, center_y = w // 2, h // 2
    else:  # full
        # Resize full frame
        resized = cv2.resize(frame, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
        return resized
    
    # Calculate crop bounds
    half_size = output_size * 2  # Crop larger area first, then resize
    x1 = max(0, center_x - half_size)
    x2 = min(w, center_x + half_size)
    y1 = max(0, center_y - half_size)
    y2 = min(h, center_y + half_size)
    
    # Crop and resize
    cropped = frame[y1:y2, x1:x2]
    if cropped.size > 0:
        resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
        return resized
    else:
        # Final fallback - just resize original
        return cv2.resize(frame, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)

def calculate_stable_crop_region(face_regions, frame_shape, crop_type="lips", output_size=96):
    """
    Calculate a stable crop region based on all face detections in the video
    This prevents jitter by using a consistent crop area for all frames
    """
    h, w = frame_shape[:2]
    
    # Filter out None detections and collect valid face regions
    valid_faces = [face for face in face_regions if face is not None]
    
    if not valid_faces:
        # No faces detected, use center crop fallback
        if crop_type == "lips":
            center_x, center_y = w // 2, int(h * 0.75)
        elif crop_type == "face":
            center_x, center_y = w // 2, h // 2
        else:  # full - now uses face-like fallback
            center_x, center_y = w // 2, h // 2
        
        crop_size = output_size * 4  # Larger crop area, will be resized
        x1 = max(0, center_x - crop_size // 2)
        y1 = max(0, center_y - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        return (x1, y1, x2 - x1, y2 - y1)
    
    # Calculate median/average face region for stability
    x_coords = [face[0] for face in valid_faces]
    y_coords = [face[1] for face in valid_faces]
    w_coords = [face[2] for face in valid_faces]
    h_coords = [face[3] for face in valid_faces]
    
    # Use median for more robust estimation
    median_x = int(np.median(x_coords))
    median_y = int(np.median(y_coords))
    median_w = int(np.median(w_coords))
    median_h = int(np.median(h_coords))
    
    # Calculate stable crop region based on median face
    if crop_type == "lips":
        # Extract lip region from median face (bottom 40% of face)
        lip_y = median_y + int(median_h * 0.6)
        lip_h = int(median_h * 0.5)  # Slightly larger for stability
        lip_x = median_x + int(median_w * 0.15)  # Center with margin
        lip_w = int(median_w * 0.7)  # Wider for stability
        
        # Ensure coordinates are within frame
        lip_x = max(0, min(lip_x, w))
        lip_y = max(0, min(lip_y, h))
        lip_w = min(lip_w, w - lip_x)
        lip_h = min(lip_h, h - lip_y)
        
        return (lip_x, lip_y, lip_w, lip_h)
        
    elif crop_type == "face":
        # Use just the detected face region with minimal padding
        padding = int(max(median_w, median_h) * 0.05)  # Very minimal padding for tight face crop
        face_x = max(0, median_x - padding)
        face_y = max(0, median_y - padding)
        face_w = min(w - face_x, median_w + 2*padding)
        face_h = min(h - face_y, median_h + 2*padding)
        
        return (face_x, face_y, face_w, face_h)
    
    else:  # full - now uses larger area around face
        # Use median face with substantial padding to cover more area
        padding = int(max(median_w, median_h) * 0.4)  # Larger padding for more coverage
        face_x = max(0, median_x - padding)
        face_y = max(0, median_y - padding)
        face_w = min(w - face_x, median_w + 2*padding)
        face_h = min(h - face_y, median_h + 2*padding)
        
        return (face_x, face_y, face_w, face_h)

def apply_stable_crop(frame, crop_region, output_size):
    """
    Apply the stable crop region to a frame and resize to output size
    """
    x, y, w, h = crop_region
    
    # Crop the frame
    cropped = frame[y:y+h, x:x+w]
    
    if cropped.size == 0:
        # Fallback to full frame if crop failed
        cropped = frame
    
    # Resize to target output size
    target_size = (output_size, output_size)
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return resized

def process_video_optimized(input_path, output_path, crop_type="lips", output_size=96, detector_type="mediapipe"):
    """
    Process HD video with stable cropping to avoid jitter
    Uses temporal smoothing and fallback strategies
    """
    cap = cv2.VideoCapture(str(input_path))
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        cap.release()
        return False
    
    # Create detector instance based on specified type
    detector_instance = get_landmarks_detector(detector_type)
    
    # Strategy 1: Try stable cropping for short videos
    if frame_count < 200:  # For short videos, use stable crop
        try:
            return process_with_stable_crop(cap, output_path, crop_type, output_size, fps, detector_instance)
        except Exception as e:
            print(f"Error in stable crop: {e}, falling back to simple method")
            # Fallback to simple method
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            return process_with_simple_crop(cap, output_path, crop_type, output_size, fps, detector_instance)
    else:
        # For longer videos, use simple method to avoid memory issues
        return process_with_simple_crop(cap, output_path, crop_type, output_size, fps, detector_instance)

def process_with_stable_crop(cap, output_path, crop_type, output_size, fps, detector_instance):
    """Process video with stable cropping (memory intensive but smooth)"""
    # First pass: collect frames and face landmarks
    all_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    
    cap.release()
    
    if not all_frames:
        return False
    
    # Detect landmarks for all frames at once for efficiency
    all_landmarks = detector_instance(all_frames)
    
    # Determine detector type for proper processing
    detector_type = "opencv"
    if hasattr(detector_instance, 'short_range_detector'):
        detector_type = "mediapipe"
    elif hasattr(detector_instance, 'landmark_detector'):
        detector_type = "retinaface"
    
    # Calculate stable crop region based on landmarks
    stable_crop_region = calculate_stable_crop_region_from_landmarks(all_landmarks, all_frames[0].shape, crop_type, output_size)
    
    # Apply stable cropping to all frames
    processed_frames = []
    for frame in all_frames:
        processed_frame = apply_stable_crop(frame, stable_crop_region, output_size)
        processed_frames.append(processed_frame)
    
    # Save video
    return save_processed_video(processed_frames, output_path, fps, output_size)

def calculate_stable_crop_region_from_landmarks(landmarks_list, frame_shape, crop_type="lips", output_size=96):
    """
    Calculate a stable crop region based on all face landmarks in the video
    This prevents jitter by using a consistent crop area for all frames
    """
    h, w = frame_shape[:2]
    
    # Filter out None detections and collect valid landmarks
    valid_landmarks = [lm for lm in landmarks_list if lm is not None]
    
    if not valid_landmarks:
        # No landmarks detected, use center crop fallback
        if crop_type == "lips":
            center_x, center_y = w // 2, int(h * 0.75)
        elif crop_type == "face":
            center_x, center_y = w // 2, h // 2
        else:  # full
            center_x, center_y = w // 2, h // 2
        
        crop_size = output_size * 4  # Larger crop area, will be resized
        x1 = max(0, center_x - crop_size // 2)
        y1 = max(0, center_y - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        return (x1, y1, x2 - x1, y2 - y1)
    
    # Determine detector type based on number of landmarks
    is_retinaface = len(valid_landmarks[0]) > 20
    is_mediapipe = len(valid_landmarks[0]) <= 5 and len(valid_landmarks[0]) > 0
    
    # Process based on crop type and detector
    if crop_type == "lips":
        if is_retinaface:
            # For RetinaFace, use mouth landmarks (indices 48-68)
            all_mouth_points_x = []
            all_mouth_points_y = []
            
            for landmarks in valid_landmarks:
                mouth_landmarks = landmarks[48:68]
                all_mouth_points_x.extend([p[0] for p in mouth_landmarks])
                all_mouth_points_y.extend([p[1] for p in mouth_landmarks])
                
            # Calculate median mouth position
            median_x = int(np.median(all_mouth_points_x))
            median_y = int(np.median(all_mouth_points_y))
            
            # Get boundaries
            min_x = min(all_mouth_points_x)
            max_x = max(all_mouth_points_x)
            min_y = min(all_mouth_points_y)
            max_y = max(all_mouth_points_y)
            
            # Calculate crop with margin
            width = max_x - min_x
            height = max_y - min_y
            
            crop_width = int(width * 2.0)  # Add margin
            crop_height = int(height * 2.5)  # More vertical margin
            
        elif is_mediapipe:
            # For MediaPipe, use the mouth point (index 3)
            mouth_points_x = [landmarks[3][0] for landmarks in valid_landmarks]
            mouth_points_y = [landmarks[3][1] for landmarks in valid_landmarks]
            
            # Get median mouth position
            median_x = int(np.median(mouth_points_x))
            median_y = int(np.median(mouth_points_y))
            
            # Calculate face size to determine lip crop size
            face_widths = []
            face_heights = []
            for landmarks in valid_landmarks:
                face_widths.append(max([p[0] for p in landmarks]) - min([p[0] for p in landmarks]))
                face_heights.append(max([p[1] for p in landmarks]) - min([p[1] for p in landmarks]))
            
            median_face_width = np.median(face_widths)
            median_face_height = np.median(face_heights)
            
            # Crop proportional to face size
            crop_width = int(median_face_width * 0.5)
            crop_height = int(median_face_height * 0.3)
            
        else:
            # For OpenCV detector or other simple landmarks
            all_x = []
            all_y = []
            
            # Collect all x and y coordinates from all valid landmarks
            for landmarks in valid_landmarks:
                for point in landmarks:
                    all_x.append(point[0])
                    all_y.append(point[1])
            
            # Calculate median point
            median_x = int(np.median(all_x))
            median_y = int(np.median(all_y))
            
            # Determine crop size based on landmarks spread
            x_min = min(all_x)
            x_max = max(all_x)
            y_min = min(all_y)
            y_max = max(all_y)
            
            width = x_max - x_min
            height = y_max - y_min
            
            # For lips, focus on lower part of face
            crop_width = int(width * 0.6)
            crop_height = int(height * 0.4)
            median_y += int(height * 0.1)  # Shift down slightly to get lips
        
        # Center the crop on the median point
        x1 = median_x - crop_width // 2
        y1 = median_y - crop_height // 2
        
    elif crop_type == "face":
        # For face detection, use all landmarks
        all_x = []
        all_y = []
        
        # Collect all coordinates from valid landmarks
        for landmarks in valid_landmarks:
            all_x.extend([p[0] for p in landmarks])
            all_y.extend([p[1] for p in landmarks])
        
        # Get median position and boundaries
        median_x = int(np.median(all_x))
        median_y = int(np.median(all_y))
        
        x_min = min(all_x)
        x_max = max(all_x)
        y_min = min(all_y)
        y_max = max(all_y)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # For face, use a wider crop
        if is_retinaface:
            # For RetinaFace, we have precise landmarks
            crop_width = int(width * 1.2)
            crop_height = int(height * 1.2)
        else:
            # For other detectors, use more margin
            crop_width = int(width * 1.5)
            crop_height = int(height * 1.5)
        
        x1 = median_x - crop_width // 2
        y1 = median_y - crop_height // 2
        
    else:  # full
        # For full, use an even wider crop
        all_x = []
        all_y = []
        
        for landmarks in valid_landmarks:
            all_x.extend([p[0] for p in landmarks])
            all_y.extend([p[1] for p in landmarks])
        
        median_x = int(np.median(all_x))
        median_y = int(np.median(all_y))
        
        x_min = min(all_x)
        x_max = max(all_x)
        y_min = min(all_y)
        y_max = max(all_y)
        
        width = x_max - x_min
        height = y_max - y_min
        
        crop_width = int(width * 3.0)
        crop_height = int(height * 3.0)
        
        x1 = median_x - crop_width // 2
        y1 = median_y - crop_height // 2
    
    # Ensure coordinates are within frame
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    crop_width = min(crop_width, w - x1)
    crop_height = min(crop_height, h - y1)
    
    return (x1, y1, crop_width, crop_height)

def process_with_simple_crop(cap, output_path, crop_type, output_size, fps, detector_instance):
    """Process video with enhanced temporal smoothing (memory efficient)"""
    processed_frames = []
    prev_crop_region = None
    smoothing_factor = 0.8  # Increased from 0.7 for more stability
    
    # Additional smoothing buffer for even more stability
    crop_history = []
    history_size = 5  # Keep track of last 5 crop regions
    
    # Determine detector type
    detector_type = "opencv"
    if hasattr(detector_instance, 'short_range_detector'):
        detector_type = "mediapipe"
    elif hasattr(detector_instance, 'landmark_detector'):
        detector_type = "retinaface"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect landmarks for current frame
        landmarks = detector_instance([frame])[0]
        
        if landmarks is not None:
            # Calculate crop region for this frame based on landmarks and detector type
            current_crop = calculate_crop_from_landmarks(landmarks, frame.shape, crop_type, detector_type)
            
            # Add to history buffer
            crop_history.append(current_crop)
            if len(crop_history) > history_size:
                crop_history.pop(0)
            
            # Use median of recent crops for extra stability
            if len(crop_history) >= 3:
                smoothed_crop = get_median_crop_region(crop_history)
            else:
                smoothed_crop = current_crop
            
            # Additional smoothing with previous region
            if prev_crop_region is not None:
                smoothed_crop = blend_crop_regions(prev_crop_region, smoothed_crop, smoothing_factor)
            
            prev_crop_region = smoothed_crop
        else:
            # No landmarks detected, use previous region or fallback
            if prev_crop_region is not None:
                smoothed_crop = prev_crop_region
            else:
                # Fallback to center crop
                smoothed_crop = get_fallback_crop(frame.shape, crop_type, output_size)
                prev_crop_region = smoothed_crop
        
        # Apply crop and add to processed frames
        processed_frame = apply_stable_crop(frame, smoothed_crop, output_size)
        processed_frames.append(processed_frame)
    
    cap.release()
    
    if not processed_frames:
        return False
    
    # Save video
    return save_processed_video(processed_frames, output_path, fps, output_size)

def calculate_crop_from_landmarks(landmarks, frame_shape, crop_type, detector_type="mediapipe"):
    """Calculate crop region from facial landmarks based on detector type"""
    h, w = frame_shape[:2]
    
    # Detect which landmark format we're dealing with
    is_retinaface = len(landmarks) > 20  # RetinaFace provides 68 landmarks
    is_mediapipe = len(landmarks) <= 5 and len(landmarks) > 0  # MediaPipe provides 4 key points
    
    if crop_type == "lips":
        if is_retinaface:
            # For RetinaFace, use the mouth landmarks (indices 48-68)
            mouth_landmarks = landmarks[48:68]
            
            # Calculate the center and boundaries of the mouth region
            mouth_x = np.mean([p[0] for p in mouth_landmarks])
            mouth_y = np.mean([p[1] for p in mouth_landmarks])
            
            # Get tight boundaries around the lips
            lip_left = min([p[0] for p in mouth_landmarks])
            lip_right = max([p[0] for p in mouth_landmarks])
            lip_top = min([p[1] for p in mouth_landmarks])
            lip_bottom = max([p[1] for p in mouth_landmarks])
            
            # Calculate width and height with margins
            lip_width = (lip_right - lip_left) * 1.7  # Add horizontal margin
            lip_height = (lip_bottom - lip_top) * 2.0  # Add vertical margin
            
            # Center the crop on the mouth
            lip_x = mouth_x - lip_width / 2
            lip_y = mouth_y - lip_height / 2
            
        elif is_mediapipe:
            # MediaPipe provides 4 points - the 4th one is the mouth
            mouth_x, mouth_y = landmarks[3]
            
            # Get face size to determine appropriate lip crop size
            face_width = max([p[0] for p in landmarks]) - min([p[0] for p in landmarks])
            face_height = max([p[1] for p in landmarks]) - min([p[1] for p in landmarks])
            
            # Create a proportional crop around the mouth point
            lip_width = face_width * 0.5
            lip_height = face_height * 0.3
            
            # Center on mouth point
            lip_x = mouth_x - lip_width / 2
            lip_y = mouth_y - lip_height / 2
            
        else:
            # For OpenCV or other detectors with simple points
            # Use the center of all points as the mouth center
            center_x = np.mean([p[0] for p in landmarks])
            center_y = np.mean([p[1] for p in landmarks])
            
            # Estimate size based on the spread of landmarks
            width = max([p[0] for p in landmarks]) - min([p[0] for p in landmarks])
            height = max([p[1] for p in landmarks]) - min([p[1] for p in landmarks])
            
            # Create a crop
            lip_width = width * 0.6
            lip_height = height * 0.4
            
            # Position it in the lower part of the detected region
            lip_x = center_x - lip_width / 2
            lip_y = center_y + height * 0.1 - lip_height / 2
        
        # Ensure coordinates are within frame
        lip_x = max(0, min(lip_x, w - 1))
        lip_y = max(0, min(lip_y, h - 1))
        lip_width = min(lip_width, w - lip_x)
        lip_height = min(lip_height, h - lip_y)
        
        return (int(lip_x), int(lip_y), int(lip_width), int(lip_height))
    
    elif crop_type == "face":
        # For face, use a wider crop based on all landmarks
        x_min = min([p[0] for p in landmarks])
        y_min = min([p[1] for p in landmarks])
        x_max = max([p[0] for p in landmarks])
        y_max = max([p[1] for p in landmarks])
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Add padding based on detector type
        if is_retinaface:
            # For RetinaFace, add less padding since we already have detailed landmarks
            padding_w = width * 0.2
            padding_h = height * 0.2
        else:
            # For other detectors, add more padding to ensure we get the full face
            padding_w = width * 0.4
            padding_h = height * 0.4
            
        face_x = max(0, x_min - padding_w)
        face_y = max(0, y_min - padding_h)
        face_w = min(w - face_x, width + 2*padding_w)
        face_h = min(h - face_y, height + 2*padding_h)
        
        return (int(face_x), int(face_y), int(face_w), int(face_h))
    
    else:  # full
        return (0, 0, w, h)

def get_median_crop_region(crop_history):
    """Calculate median crop region from history for extra stability"""
    if not crop_history:
        return None
    
    # Extract coordinates from all crop regions
    x_coords = [crop[0] for crop in crop_history]
    y_coords = [crop[1] for crop in crop_history]
    w_coords = [crop[2] for crop in crop_history]
    h_coords = [crop[3] for crop in crop_history]
    
    # Calculate median values
    median_x = int(np.median(x_coords))
    median_y = int(np.median(y_coords))
    median_w = int(np.median(w_coords))
    median_h = int(np.median(h_coords))
    
    return (median_x, median_y, median_w, median_h)

def blend_crop_regions(prev_region, current_region, smoothing_factor):
    """Blend two crop regions for temporal smoothing"""
    px, py, pw, ph = prev_region
    cx, cy, cw, ch = current_region
    
    # Weighted average
    alpha = smoothing_factor
    blended_x = int(alpha * px + (1 - alpha) * cx)
    blended_y = int(alpha * py + (1 - alpha) * cy)
    blended_w = int(alpha * pw + (1 - alpha) * cw)
    blended_h = int(alpha * ph + (1 - alpha) * ch)
    
    return (blended_x, blended_y, blended_w, blended_h)

def get_fallback_crop(frame_shape, crop_type, output_size):
    """Get fallback crop region when no face is detected"""
    h, w = frame_shape[:2]
    
    if crop_type == "lips":
        center_x, center_y = w // 2, int(h * 0.75)
    elif crop_type == "face":
        center_x, center_y = w // 2, h // 2
    else:  # full
        return (0, 0, w, h)
    
    crop_size = output_size * 4
    x1 = max(0, center_x - crop_size // 2)
    y1 = max(0, center_y - crop_size // 2)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    
    return (x1, y1, x2 - x1, y2 - y1)

def save_processed_video(processed_frames, output_path, fps, output_size):
    """Save processed frames to video file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_size, output_size))
    
    for frame in processed_frames:
        out.write(frame)
    
    out.release()
    return True

def extract_audio(input_video_path, output_audio_path):
    """
    Extract audio from video using ffmpeg
    Returns True if successful, False otherwise
    """
    try:
        # Use ffmpeg to extract audio as WAV file
        cmd = [
            'ffmpeg', '-i', str(input_video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate (common for speech)
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            str(output_audio_path)
        ]
        
        # Run ffmpeg with suppressed output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            return False
            
    except Exception:
        return False

def find_video_files(data_dir, subset):
    """Find all video files in the subset directory"""
    subset_path = Path(data_dir) / subset
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(subset_path.rglob(f"*{ext}"))
    
    return sorted(video_files)

def extract_file_info(video_path, subset_dir):
    """Extract speaker ID and transcript ID from video path"""
    rel_path = video_path.relative_to(subset_dir)
    parts = rel_path.parts
    # Expecting: speaker/Clips/camera_view/filename
    if len(parts) >= 4:
        speaker_id = parts[0]
        session = parts[1]  # e.g., 'Clips'
        camera_view = parts[2]  # e.g., '30degcam', 'frontalcam'
        video_name = parts[-1]
        transcript_id = video_name.split('.')[0]
        return speaker_id, session, camera_view, transcript_id
    elif len(parts) >= 2:
        # Fallback: just speaker and filename
        speaker_id = parts[0]
        session = None
        camera_view = None
        video_name = parts[-1]
        transcript_id = video_name.split('.')[0]
        return speaker_id, session, camera_view, transcript_id
    return None, None, None, None

def main():
    parser = argparse.ArgumentParser(description="TCD-TIMIT Preprocessing Pipeline (HD-Optimized)")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory of TCD-TIMIT dataset")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory for output")
    parser.add_argument("--subset", type=str, required=True, choices=["volunteers", "lipspeakers"], 
                       help="Subset to process")
    parser.add_argument("--crop-type", type=str, default="lips",
                       choices=["lips", "face", "full"], help="Crop type")
    parser.add_argument("--output-size", type=int, default=96,
                       help="Output video resolution (e.g., 96 for 96x96, 224 for 224x224)")
    parser.add_argument("--detector", type=str, default="mediapipe",
                       choices=["mediapipe", "retinaface", "opencv"], 
                       help="Face detector to use (mediapipe is recommended, retinaface requires GPU)")
    parser.add_argument("--max-videos", type=int, default=None,
                       help="Maximum number of videos to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate output size
    if args.crop_type == "lips" and args.output_size > 128:
        print("⚠️  Warning: Large output size for lips. Consider 96x96 for efficiency.")
    elif args.crop_type == "face" and args.output_size > 256:
        print("⚠️  Warning: Large output size for face. Consider 224x224 for efficiency.")
    
    # Display configuration
    print("🚀 Starting TCD-TIMIT preprocessing...")
    print(f"📁 Output: {args.root_dir}")
    print(f"🎥 Crop: {args.crop_type} ({args.output_size}x{args.output_size})")
    print(f"👥 Subset: {args.subset}")
    print(f"🔍 Detector: {args.detector}")
    print("-" * 50)
    
    # Setup output directories
    dataset = "tcd_timit"
    seg_duration = 16
    size_suffix = f"_{args.output_size}x{args.output_size}"
    crop_suffix = f"_{args.crop_type}" if args.crop_type != "lips" else ""
    
    dst_vid_dir = os.path.join(args.root_dir, dataset, f"{dataset}_video_seg{seg_duration}s{crop_suffix}{size_suffix}")
    dst_txt_dir = os.path.join(args.root_dir, dataset, f"{dataset}_text_seg{seg_duration}s{crop_suffix}")
    dst_aud_dir = dst_vid_dir  # Audio files in same directory as video files
    labels_dir = os.path.join(args.root_dir, dataset, "labels")
    

    
    # Find transcript files
    data_path = Path(args.data_dir)
    
    # The MLF files are in the root data directory
    subset_mapping = {
        "volunteers": "volunteer",
        "lipspeakers": "lipspeaker"
    }
    
    mlf_prefix = subset_mapping.get(args.subset, args.subset)
    mlf_filename = f"{mlf_prefix}_labelfiles.mlf"
    mlf_file = data_path / mlf_filename
    
    if not mlf_file.exists():
        print(f"❌ Error: MLF file not found at {mlf_file}")
        return 1
    
    # Parse the transcript file
    all_transcripts = parse_mlf_transcripts(mlf_file)
    
    # Find video files
    video_files = find_video_files(args.data_dir, args.subset)
    if not video_files:
        print("❌ Error: No video files found")
        return 1
    
    # Limit for testing if specified
    if args.max_videos:
        video_files = video_files[:args.max_videos]
    
    # Group by (speaker, session, camera_view)
    speaker_videos = {}
    subset_dir = data_path / args.subset

    for video_path in video_files:
        speaker_id, session, camera_view, transcript_id = extract_file_info(video_path, subset_dir)
        if speaker_id and transcript_id:
            key = (speaker_id, session, camera_view)
            if key not in speaker_videos:
                speaker_videos[key] = []
            speaker_videos[key].append((video_path, transcript_id))

    print(f"� Found {len(video_files)} videos, {len(speaker_videos)} (speaker, session, camera_view) groups")
    
    
    # Process videos
    csv_data = []
    processed_count = 0
    skipped_count = 0
    
    for (speaker_id, session, camera_view) in tqdm(speaker_videos.keys(), desc="Processing speakers/sessions/views"):
        videos = speaker_videos[(speaker_id, session, camera_view)]

        # Create output directories including session and camera_view
        if session and camera_view:
            speaker_vid_dir = os.path.join(dst_vid_dir, args.subset, speaker_id, session, camera_view)
            speaker_txt_dir = os.path.join(dst_txt_dir, args.subset, speaker_id, session, camera_view)
            speaker_aud_dir = os.path.join(dst_aud_dir, args.subset, speaker_id, session, camera_view)
        else:
            speaker_vid_dir = os.path.join(dst_vid_dir, args.subset, speaker_id)
            speaker_txt_dir = os.path.join(dst_txt_dir, args.subset, speaker_id)
            speaker_aud_dir = os.path.join(dst_aud_dir, args.subset, speaker_id)
        os.makedirs(speaker_vid_dir, exist_ok=True)
        os.makedirs(speaker_txt_dir, exist_ok=True)
        os.makedirs(speaker_aud_dir, exist_ok=True)

        for video_path, transcript_id in tqdm(videos, desc=f"  {speaker_id}/{session}/{camera_view}", leave=False):
            # Check if transcript exists
            if transcript_id not in all_transcripts:
                skipped_count += 1
                continue

            transcript = all_transcripts[transcript_id]

            # Create unique file ID: speaker_session_camera_transcript (e.g., 01M_Clips_30degcam_sa1)
            if session and camera_view:
                unique_id = f"{speaker_id}_{session}_{camera_view}_{transcript_id}"
            else:
                unique_id = f"{speaker_id}_{transcript_id}"

            # Output paths with unique naming - put audio in same dir as video for easier processing
            output_video = os.path.join(speaker_vid_dir, f"{unique_id}.mp4")
            output_phn = os.path.join(speaker_txt_dir, f"{unique_id}.phn")    # Phonemes
            output_txt = os.path.join(speaker_txt_dir, f"{unique_id}.txt")    # Sentence
            output_audio = os.path.join(speaker_vid_dir, f"{unique_id}.wav")  # Same dir as video

            try:
                # Process video with optimized pipeline
                success = process_video_optimized(video_path, output_video, args.crop_type, args.output_size, args.detector)
                if not success:
                    print(f"⚠️  Warning: Failed to process {video_path}")
                    skipped_count += 1
                    continue

                # Extract audio
                extract_audio(video_path, output_audio)

                # Save phonemes (.phn file)
                with open(output_phn, 'w') as f:
                    f.write(transcript)  # This is the phoneme sequence

                # Save sentence (.txt file) if we have a mapping
                sentence = TIMIT_SENTENCES.get(transcript_id, transcript)  # Fallback to phonemes if no sentence mapping
                with open(output_txt, 'w') as f:
                    f.write(sentence)

                # Add to CSV data
                if session and camera_view:
                    rel_video_path = f"{dataset}_video_seg{seg_duration}s{crop_suffix}{size_suffix}/{args.subset}/{speaker_id}/{session}/{camera_view}/{unique_id}.mp4"
                else:
                    rel_video_path = f"{dataset}_video_seg{seg_duration}s{crop_suffix}{size_suffix}/{args.subset}/{speaker_id}/{unique_id}.mp4"
                csv_data.append([
                    speaker_id,
                    rel_video_path,
                    sentence,
                    len(sentence.split()),  # word count
                    unique_id,  # Use unique ID instead of transcript_id
                    f"{args.output_size}x{args.output_size}",  # resolution
                    args.crop_type
                ])

                processed_count += 1

            except Exception as e:
                print(f"❌ Error processing {video_path}: {e}")
                skipped_count += 1
                continue
    
    # Save CSV
    if csv_data:
        os.makedirs(labels_dir, exist_ok=True)
        csv_filename = f"tcd_timit_{args.subset}{crop_suffix}{size_suffix}.csv"
        csv_path = os.path.join(labels_dir, csv_filename)
        
        df = pd.DataFrame(csv_data, columns=[
            'speaker_id', 'video_path', 'transcript', 'word_count', 
            'transcript_id', 'resolution', 'crop_type'
        ])
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Saved: {csv_path}")
        print(f"📊 Total: {len(df)} samples ({args.output_size}x{args.output_size}, {args.crop_type})")

    print(f"\n✅ Complete! Processed: {processed_count}, Skipped: {skipped_count}")
    return 0

if __name__ == "__main__":
    exit(main())
