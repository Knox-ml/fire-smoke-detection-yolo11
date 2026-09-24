"""
Fire & Smoke Detection — Gradio Demo
=====================================
Deploy on HuggingFace Spaces for a live demo link in your README.

Setup on HF Spaces:
  1. Create new Space (Gradio SDK)
  2. Upload: app.py, requirements.txt, best.pt
  3. Done — it auto-deploys
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model (place best.pt in the same directory)
model = YOLO("best.pt")

CLASS_COLORS = {
    0: (255, 60, 60),    # Fire — red
    1: (180, 180, 180),  # Smoke — gray
}


def detect_fire(input_image: Image.Image, confidence: float = 0.25):
    """Run fire/smoke detection on uploaded image."""
    # Convert PIL to numpy
    img = np.array(input_image)

    # Run inference
    results = model(img, conf=confidence, verbose=False)
    result = results[0]

    # Get annotated image
    annotated = result.plot(
        conf=True,
        line_width=2,
        font_size=12,
    )

    # Build detection summary
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = result.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        detections.append(f"{cls_name}: {conf:.2%} at [{x1}, {y1}, {x2}, {y2}]")

    if not detections:
        summary = "No fire or smoke detected."
    else:
        summary = f"Found {len(detections)} detection(s):\n" + "\n".join(detections)

    return Image.fromarray(annotated), summary


def detect_fire_video(input_video: str, confidence: float = 0.25):
    """Run fire/smoke detection on uploaded video."""
    cap = cv2.VideoCapture(input_video)

    # Output setup
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    fire_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence, verbose=False)
        annotated = results[0].plot(line_width=2)
        writer.write(annotated)

        if len(results[0].boxes) > 0:
            fire_frames += 1
        frame_count += 1

    cap.release()
    writer.release()

    summary = (
        f"Processed {frame_count} frames.\n"
        f"Fire/smoke detected in {fire_frames} frames "
        f"({fire_frames / max(frame_count, 1) * 100:.1f}%)."
    )

    return output_path, summary


# ============================================================
# Gradio Interface
# ============================================================
with gr.Blocks(
    title="Fire & Smoke Detection — YOLOv8",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 🔥 Fire & Smoke Detection — YOLOv8
        Real-time fire and smoke detection using YOLOv8 fine-tuned on the
        [D-Fire dataset](https://github.com/gaiasd/DFireDataset) (21K+ images).

        Upload an image or video to detect fire and smoke.
        """
    )

    with gr.Tab("Image Detection"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Upload Image")
                conf_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                    label="Confidence Threshold",
                )
                img_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                img_output = gr.Image(type="pil", label="Detection Result")
                img_summary = gr.Textbox(label="Detection Summary", lines=5)

        img_btn.click(
            detect_fire,
            inputs=[img_input, conf_slider],
            outputs=[img_output, img_summary],
        )

    with gr.Tab("Video Detection"):
        with gr.Row():
            with gr.Column():
                vid_input = gr.Video(label="Upload Video")
                vid_conf = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                    label="Confidence Threshold",
                )
                vid_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                vid_output = gr.Video(label="Detection Result")
                vid_summary = gr.Textbox(label="Summary", lines=3)

        vid_btn.click(
            detect_fire_video,
            inputs=[vid_input, vid_conf],
            outputs=[vid_output, vid_summary],
        )

    gr.Markdown(
        """
        ---
        **Model:** YOLOv8 fine-tuned on D-Fire | **Classes:** Fire, Smoke |
        **Dataset:** 21K+ images | [GitHub Repo](https://github.com/YOUR_USERNAME/fire-detection-yolov8)
        """
    )

if __name__ == "__main__":
    demo.launch()
