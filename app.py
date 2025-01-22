import os
from typing import List
import spaces
import gradio as gr
import numpy as np
import torch
import json
import tempfile
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from gradio.themes.utils import sizes
from classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    GOLIATH_KEYPOINTS
)

import os
import sys
import subprocess
import importlib.util

def is_package_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

def find_wheel(package_path):
    dist_dir = os.path.join(package_path, "dist")
    if os.path.exists(dist_dir):
        wheel_files = [f for f in os.listdir(dist_dir) if f.endswith('.whl')]
        if wheel_files:
            return os.path.join(dist_dir, wheel_files[0])
    return None

def install_from_wheel(package_name, package_path):
    wheel_file = find_wheel(package_path)
    if wheel_file:
        print(f"Installing {package_name} from wheel: {wheel_file}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_file])
    else:
        print(f"{package_name} wheel not found in {package_path}. Please build it first.")
        sys.exit(1)

def install_local_packages():
    packages = [
        ("mmengine", "./external/engine"),
        ("mmcv", "./external/cv"),
        ("mmdet", "./external/det")
    ]
    
    for package_name, package_path in packages:
        if not is_package_installed(package_name):
            print(f"Installing {package_name}...")
            install_from_wheel(package_name, package_path)
        else:
            print(f"{package_name} is already installed.")

# Run the installation at the start of your app
install_local_packages()

from detector_utils import (
            adapt_mmdet_pipeline,
            init_detector,
            process_images_detector,
        )

class Config:
    ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
    CHECKPOINTS_DIR = os.path.join(ASSETS_DIR, "checkpoints")
    CHECKPOINTS = {
        "0.3b": "sapiens_0.3b_goliath_best_goliath_AP_575_torchscript.pt2",
        "0.6b": "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2",
        "1b": "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2",
    }
    DETECTION_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth')
    DETECTION_CONFIG = os.path.join(ASSETS_DIR, 'rtmdet_m_640-8xb32_coco-person_no_nms.py')

class ModelManager:
    @staticmethod
    def load_model(checkpoint_name: str):
        if checkpoint_name is None:
            return None
        checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, checkpoint_name)
        model = torch.jit.load(checkpoint_path)
        model.eval()
        model.to("cuda")
        return model

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor):
        return model(input_tensor)

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], 
                                 std=[58.5/255, 57.0/255, 57.5/255])
        ])
        self.detector = init_detector(
            Config.DETECTION_CONFIG, Config.DETECTION_CHECKPOINT, device='cpu'
        )
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

    def detect_persons(self, image: Image.Image):
        # Convert PIL Image to tensor
        image = np.array(image)
        image = np.expand_dims(image, axis=0)

        # Perform person detection
        bboxes_batch = process_images_detector(
            image, 
            self.detector
        )
        bboxes = self.get_person_bboxes(bboxes_batch[0])  # Get bboxes for the first (and only) image
        
        return bboxes
    
    def get_person_bboxes(self, bboxes_batch, score_thr=0.3):
        person_bboxes = []
        for bbox in bboxes_batch:
            if len(bbox) == 5:  # [x1, y1, x2, y2, score]
                if bbox[4] > score_thr:
                    person_bboxes.append(bbox)
            elif len(bbox) == 4:  # [x1, y1, x2, y2]
                person_bboxes.append(bbox + [1.0])  # Add a default score of 1.0
        return person_bboxes

    @spaces.GPU
    @torch.inference_mode()
    def estimate_pose(self, image: Image.Image, bboxes: List[List[float]], model_name: str, kpt_threshold: float):
        pose_model = ModelManager.load_model(Config.CHECKPOINTS[model_name])
        
        result_image = image.copy()
        all_keypoints = []  # List to store keypoints for all persons

        for bbox in bboxes:
            cropped_img = self.crop_image(result_image, bbox)
            input_tensor = self.transform(cropped_img).unsqueeze(0).to("cuda")
            heatmaps = ModelManager.run_model(pose_model, input_tensor)
            keypoints = self.heatmaps_to_keypoints(heatmaps[0].cpu().numpy(), bbox)
            all_keypoints.append(keypoints)  # Collect keypoints
            result_image = self.draw_keypoints(result_image, keypoints, bbox, kpt_threshold)
        
        return result_image, all_keypoints

    def process_image(self, image: Image.Image, model_name: str, kpt_threshold: str):
        bboxes = self.detect_persons(image)
        result_image, keypoints = self.estimate_pose(image, bboxes, model_name, float(kpt_threshold))
        return result_image, keypoints

    def crop_image(self, image, bbox):
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
        elif len(bbox) >= 5:
            x1, y1, x2, y2, _ = map(int, bbox[:5])
        else:
            raise ValueError(f"Unexpected bbox format: {bbox}")
        
        crop = image.crop((x1, y1, x2, y2))
        return crop

    @staticmethod
    def heatmaps_to_keypoints(heatmaps, bbox):
        num_joints = heatmaps.shape[0]  # Should be 308
        keypoints = {}
        x1, y1, x2, y2 = map(int, bbox[:4])
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        for i, name in enumerate(GOLIATH_KEYPOINTS):
            if i < num_joints:
                heatmap = heatmaps[i]
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                conf = heatmap[y, x]
                # Convert coordinates to image frame
                x_image = x * bbox_width / 192 + x1
                y_image = y * bbox_height / 256 + y1
                keypoints[name] = (float(x_image), float(y_image), float(conf))
        return keypoints

    @staticmethod
    def draw_keypoints(image, keypoints, bbox, kpt_threshold):
        image = np.array(image)

        # Handle both 4 and 5-element bounding boxes
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
        elif len(bbox) >= 5:
            x1, y1, x2, y2, _ = map(int, bbox[:5])
        else:
            raise ValueError(f"Unexpected bbox format: {bbox}")
                
        # Calculate adaptive radius and thickness based on bounding box size
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_size = np.sqrt(bbox_width * bbox_height)
        
        radius = max(1, int(bbox_size * 0.006))  # minimum 1 pixel
        thickness = max(1, int(bbox_size * 0.006))  # minimum 1 pixel
        bbox_thickness = max(1, thickness//4)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), bbox_thickness)
        
        # Draw keypoints
        for i, (name, (x, y, conf)) in enumerate(keypoints.items()):
            if conf > kpt_threshold and i < len(GOLIATH_KPTS_COLORS):
                x_coord = int(x)
                y_coord = int(y)
                color = GOLIATH_KPTS_COLORS[i]
                cv2.circle(image, (x_coord, y_coord), radius, color, -1)

        # Draw skeleton
        for _, link_info in GOLIATH_SKELETON_INFO.items():
            pt1_name, pt2_name = link_info['link']
            color = link_info['color']
            
            if pt1_name in keypoints and pt2_name in keypoints:
                pt1 = keypoints[pt1_name]
                pt2 = keypoints[pt2_name]
                if pt1[2] > kpt_threshold and pt2[2] > kpt_threshold:
                    x1_coord = int(pt1[0])
                    y1_coord = int(pt1[1])
                    x2_coord = int(pt2[0])
                    y2_coord = int(pt2[1])
                    cv2.line(image, (x1_coord, y1_coord), (x2_coord, y2_coord), color, thickness=thickness)

        return Image.fromarray(image)

class GradioInterface:
    def __init__(self):
        self.image_processor = ImageProcessor()

    def create_interface(self):
        app_styles = """
        <style>
            /* Global Styles */
            body, #root {
                font-family: Helvetica, Arial, sans-serif;
                background-color: #1a1a1a;
                color: #fafafa;
            }
            /* Header Styles */
            .app-header {
                background: linear-gradient(45deg, #1a1a1a 0%, #333333 100%);
                padding: 24px;
                border-radius: 8px;
                margin-bottom: 24px;
                text-align: center;
            }
            .app-title {
                font-size: 48px;
                margin: 0;
                color: #fafafa;
            }
            .app-subtitle {
                font-size: 24px;
                margin: 8px 0 16px;
                color: #fafafa;
            }
            .app-description {
                font-size: 16px;
                line-height: 1.6;
                opacity: 0.8;
                margin-bottom: 24px;
            }
            /* Button Styles */
            .publication-links {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 16px;
            }
            .publication-link {
                display: inline-flex;
                align-items: center;
                padding: 8px 16px;
                background-color: #333;
                color: #fff !important;
                text-decoration: none !important;
                border-radius: 20px;
                font-size: 14px;
                transition: background-color 0.3s;
            }
            .publication-link:hover {
                background-color: #555;
            }
            .publication-link i {
                margin-right: 8px;
            }
            /* Content Styles */
            .content-container {
                background-color: #2a2a2a;
                border-radius: 8px;
                padding: 24px;
                margin-bottom: 24px;
            }
            /* Image Styles */
            .image-preview img {
                max-width: 512px;
                max-height: 512px;
                margin: 0 auto;
                border-radius: 4px;
                display: block;
                object-fit: contain;  
            }
            /* Control Styles */
            .control-panel {
                background-color: #333;
                padding: 16px;
                border-radius: 8px;
                margin-top: 16px;
            }
            /* Gradio Component Overrides */
            .gr-button {
                background-color: #4a4a4a;
                color: #fff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .gr-button:hover {
                background-color: #5a5a5a;
            }
            .gr-input, .gr-dropdown {
                background-color: #3a3a3a;
                color: #fff;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 8px;
            }
            .gr-form {
                background-color: transparent;
            }
            .gr-panel {
                border: none;
                background-color: transparent;
            }
            /* Override any conflicting styles from Bulma */
            .button.is-normal.is-rounded.is-dark {
                color: #fff !important;
                text-decoration: none !important;
            }
        </style>
        """

        header_html = f"""
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.3/css/bulma.min.css">
        <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.15.4/css/all.css">
        {app_styles}
        <div class="app-header">
            <h1 class="app-title">Sapiens: Pose Estimation</h1>
            <h2 class="app-subtitle">ECCV 2024 (Oral)</h2>
            <p class="app-description">
                Meta presents Sapiens, foundation models for human tasks pretrained on 300 million human images. 
                This demo showcases the finetuned pose estimation model. <br>
            </p>
            <div class="publication-links">
                <a href="https://arxiv.org/abs/2408.12569" class="publication-link">
                    <i class="fas fa-file-pdf"></i>arXiv
                </a>
                <a href="https://github.com/facebookresearch/sapiens" class="publication-link">
                    <i class="fab fa-github"></i>Code
                </a>
                <a href="https://about.meta.com/realitylabs/codecavatars/sapiens/" class="publication-link">
                    <i class="fas fa-globe"></i>Meta
                </a>
                <a href="https://rawalkhirodkar.github.io/sapiens" class="publication-link">
                    <i class="fas fa-chart-bar"></i>Results
                </a>
            </div>
            <div class="publication-links">
                <a href="https://huggingface.co/spaces/facebook/sapiens_pose" class="publication-link">
                    <i class="fas fa-user"></i>Demo-Pose
                </a>
                <a href="https://huggingface.co/spaces/facebook/sapiens_seg" class="publication-link">
                    <i class="fas fa-puzzle-piece"></i>Demo-Seg
                </a>
                <a href="https://huggingface.co/spaces/facebook/sapiens_depth" class="publication-link">
                    <i class="fas fa-cube"></i>Demo-Depth
                </a>
                <a href="https://huggingface.co/spaces/facebook/sapiens_normal" class="publication-link">
                    <i class="fas fa-vector-square"></i>Demo-Normal
                </a>
            </div>
        </div>
        """

        js_func = """
        function refresh() {
            const url = new URL(window.location);
            if (url.searchParams.get('__theme') !== 'dark') {
                url.searchParams.set('__theme', 'dark');
                window.location.href = url.href;
            }
        }
        """

        def process_image(image, model_name, kpt_threshold):
            result_image, keypoints = self.image_processor.process_image(image, model_name, kpt_threshold)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w', dir='/tmp/gradio') as json_file:
                json.dump(keypoints, json_file)
                json_file_path = json_file.name
            return result_image, json_file_path

        with gr.Blocks(js=js_func, theme=gr.themes.Default()) as demo:
            gr.HTML(header_html)
            with gr.Row(elem_classes="content-container"):
                with gr.Column():
                    input_image = gr.Image(label="Input Image", type="pil", format="png", elem_classes="image-preview")
                    with gr.Row():
                        model_name = gr.Dropdown(
                            label="Model Size",
                            choices=list(Config.CHECKPOINTS.keys()),
                            value="1b",
                        )
                        kpt_threshold = gr.Dropdown(
                            label="Min Keypoint Confidence",
                            choices=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
                            value="0.3",
                        )
                    example_model = gr.Examples(
                        inputs=input_image,
                        examples_per_page=14,
                        examples=[
                            os.path.join(Config.ASSETS_DIR, "images", img)
                            for img in os.listdir(os.path.join(Config.ASSETS_DIR, "images"))
                        ],
                    )
                with gr.Column():
                    result_image = gr.Image(label="Pose-308 Result", type="pil", elem_classes="image-preview")
                    json_output = gr.File(label="Pose-308 Output (.json)")
                    run_button = gr.Button("Run")

            run_button.click(
                fn=process_image,
                inputs=[input_image, model_name, kpt_threshold],
                outputs=[result_image, json_output],
            )
            
        return demo

def main():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    interface = GradioInterface()
    demo = interface.create_interface()
    demo.launch(share=False)

if __name__ == "__main__":
    main()
