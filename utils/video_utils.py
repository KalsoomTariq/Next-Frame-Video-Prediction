'''
    DEEP LEARNING PROJECT FOR NEXT FRAME VIDEO PREDICTION

Members:
- Kalsoom Tariq (i212487)
- Abtaal Aatif (i212990)
- Ali Ashraf (i210756))

'''

from moviepy.editor import ImageSequenceClip
import numpy as np
import torch
import cv2
import os

def load_and_preprocess_video(video_path, input_frames=10, output_frames=5):
    """
    Load video and preprocess frames
    
    Args:
        video_path (str): Path to the video file
        input_frames (int): Number of input frames to extract
        output_frames (int): Number of output frames to predict
    
    Returns:
        tuple: (input_frames, ground_truth_frames)
    """
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Collect all frames
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to 64x64
        frame_resized = cv2.resize(frame_gray, (64, 64), interpolation=cv2.INTER_AREA)
        
        all_frames.append(frame_resized)
    
    cap.release()
    
    # Ensure enough frames
    if len(all_frames) < input_frames + output_frames:
        raise ValueError(f"Not enough frames in video. Found {len(all_frames)}, need {input_frames + output_frames}")
    
    # Extract input and ground truth frames
    input_frames_data = all_frames[:input_frames]
    ground_truth_frames = all_frames[input_frames:input_frames+output_frames]
    
    # Convert to numpy arrays
    input_frames_array = np.array(input_frames_data)
    ground_truth_array = np.array(ground_truth_frames)
    
    return input_frames_array, ground_truth_array

def normalize_video(video):
    """
    Normalize video frames
    
    Args:
        video (np.ndarray): Input video frames
    
    Returns:
        torch.Tensor: Normalized frames
    """
    # Normalize to [0, 1] range
    video = video.astype(np.float32) / 255.0
    
    # Add channel dimension for grayscale
    video = video[..., np.newaxis]
    
    # Convert to torch tensor and rearrange dimensions
    return torch.from_numpy(video).permute(0, 3, 1, 2)

def save_side_by_side_video(ground_truth, prediction, output_path):
    """
    Save ground truth and prediction side by side using moviepy
    
    Args:
        ground_truth (np.ndarray): Ground truth video frames, shape (T, H, W) or (T, H, W, C)
        prediction (np.ndarray): Predicted video frames, shape (T, H, W) or (T, H, W, C)
        output_path (str): Path to save the combined video
    """
    # Ensure both videos have the same number of frames
    min_frames = min(len(ground_truth), len(prediction))
    ground_truth = ground_truth[:min_frames]
    prediction = prediction[:min_frames]

    # Ensure they are 3-channel RGB
    def prepare_frames(frames):
        if frames.ndim == 3:  # If grayscale (T, H, W), add a channel dimension
            frames = np.expand_dims(frames, axis=-1)
        if frames.shape[-1] == 1:  # If single-channel, repeat to make RGB
            frames = np.repeat(frames, 3, axis=-1)
        return frames

    ground_truth_rgb = prepare_frames(ground_truth)
    prediction_rgb = prepare_frames(prediction)

    # Combine frames side by side
    combined_frames = []
    for gt_frame, pred_frame in zip(ground_truth_rgb, prediction_rgb):
        combined_frame = np.hstack((gt_frame, pred_frame))  # Combine horizontally
        combined_frames.append(combined_frame)

    # Create video using ImageSequenceClip
    clip = ImageSequenceClip(combined_frames, fps=30)  # Specify fps for video
    clip.write_videofile(output_path, codec="libx264")  # Save video

    print(f"Video saved to {output_path}")


def get_available_classes(base_dir):
    """
    Get list of available video classes
    
    Args:
        base_dir (str): Base directory containing video classes
    
    Returns:
        list: List of available class names
    """
    return [d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))]

def get_videos_in_class(base_dir, class_name):
    """
    Get list of videos in a specific class
    
    Args:
        base_dir (str): Base directory containing video classes
        class_name (str): Name of the class
    
    Returns:
        list: List of video file names
    """
    class_path = os.path.join(base_dir, class_name)
    return [f for f in os.listdir(class_path) if f.endswith('.avi')]