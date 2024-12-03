'''
    DEEP LEARNING PROJECT FOR NEXT FRAME VIDEO PREDICTION

Members:
- Kalsoom Tariq (i212487)
- Abtaal Aatif (i212990)
- Ali Ashraf (i210756))

'''
import gradio as gr
import numpy as np
import os

# Import custom utility functions
from utils.video_utils import (
    load_and_preprocess_video, 
    normalize_video, 
    save_side_by_side_video,
    get_available_classes, 
    get_videos_in_class
)

# Import model loader
from utils.modal_loader import ModelLoader
model_loader = ModelLoader()

# Configuration
BASE_VIDEO_DIR = 'UCF101/test'
OUTPUT_VIDEO_DIR = 'output_videos'
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

def predict_video(model_name, class_name, video_name):
    """
    Main prediction function for Gradio interface
    
    Args:
        model_name (str): Name of the model to use
        class_name (str): Selected video class
        video_name (str): Selected video file name
    
    Returns:
        str: Path to side-by-side output video
    """
    # Construct full video path
    video_path = os.path.join(BASE_VIDEO_DIR, class_name, video_name)
    
    if model_name == 'ConvLSTM':
        # Load video
        input_frames, gt_frames = load_and_preprocess_video(video_path)
        
        # Normalize and convert to tensor
        input_video = normalize_video(input_frames)
        
        # Add batch and time dimensions if not present
        if len(input_video.shape) == 4:
            input_video = input_video.unsqueeze(0)
        
        # Predict next 5 frames
        predictions = model_loader.predict(model_name, input_video)
        
        # Save side-by-side video
        output_path = os.path.join(
            OUTPUT_VIDEO_DIR, 
            f'{model_name}_{class_name}_{video_name.replace(".avi", ".mp4")}'
        )

        # Perform dimensionality adjustments to create side-by-side frames
        input_frames = np.expand_dims(input_frames, axis=-1)
        gt_frames = np.expand_dims(gt_frames, axis=-1)
        org_v = np.concatenate((input_frames, gt_frames), axis=0)
        pre_v = np.concatenate((input_frames, predictions[0]), axis=0)

        # save video
        save_side_by_side_video(org_v, pre_v, output_path)

        # Return output path
        return output_path

    elif model_name == 'PredRNN':
        pass
    elif model_name == 'Transformer':
        pass

    return ''

def create_interface():
    """
    Create Gradio interface
    
    Returns:
        gr.Interface: Configured Gradio interface
    """
    # Get available classes
    classes = get_available_classes(BASE_VIDEO_DIR)
    
    with gr.Blocks(title="Video Next Frame Prediction") as demo:
        gr.Markdown("# Video Next Frame Prediction")
        gr.Markdown("Select a model, class, and video to generate predictions")
        
        with gr.Row():
            # Model selection dropdown
            model_dropdown = gr.Dropdown(
                choices=['ConvLSTM', 'PredRNN', 'Transformer'], 
                label="Select Model",
                value='ConvLSTM'
            )
            
            # Class selection dropdown
            class_dropdown = gr.Dropdown(
                choices=classes, 
                label="Select Video Class",
                value=classes[0]
            )
        
        # Video selection dropdown
        video_dropdown = gr.Dropdown(label="Select Video")
        
        # Update video dropdown when class changes
        class_dropdown.change(
            fn=lambda class_name: gr.Dropdown(
                choices=get_videos_in_class(BASE_VIDEO_DIR, class_name), 
                label="Select Video"
            ),
            inputs=class_dropdown,
            outputs=video_dropdown
        )
        
        # Predict button
        predict_btn = gr.Button("Generate Prediction", variant="primary")
        
        # Output video display
        output_video = gr.Video(label="Ground Truth vs Prediction")
        
        # Prediction workflow
        predict_btn.click(
            fn=predict_video,
            inputs=[model_dropdown, class_dropdown, video_dropdown],
            outputs=output_video
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(debug=True)