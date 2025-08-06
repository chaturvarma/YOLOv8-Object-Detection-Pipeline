from ultralytics import YOLO
import cv2
import argparse
import yaml

# Load the config file
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Load the parameters from the config file
def get_inference_params(config):
    return config["inference"]["conf_threshold"], config["inference"]["iou_threshold"]
    
config = load_config()
model_path = config["model"]
CONF_THRESHOLD, IOU_THRESHOLD = get_inference_params(config)

# Load the model
model = YOLO(model_path)

#========================================
# Object detection on a single image file
#========================================
def inference_image(image_path, output_path):
    # Perform prediction on the image
    results = model.predict(image_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    result = results[0]

    # Generate an image with bounding boxes and labels
    pred_img = result.plot()

    cv2.imwrite(output_path, pred_img)

#========================================
# Object detection on a single video file
#========================================
def inference_video(video_path, output_path):
    capture = cv2.VideoCapture(video_path)

    width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = capture.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter object to save output
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ret = True
    while ret:
        ret, frame = capture.read()
        if not ret:
            break

        # Run YOLO tracking on the current frame
        results = model.track(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, persist=True)

        result = results[0]
        pred_img = frame.copy()

        # Draw bounding boxes and labels if detections are present
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw rectangle around detected object
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Put label text above the bounding box
                cv2.putText(pred_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        writer.write(pred_img)

    capture.release()
    writer.release()

#===================================
# Object detection on webcam footage
#===================================
def inference_webcam():
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Error: Could not start webcam")
        return

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break

        # Run YOLO tracking on the webcam frame
        results = model.track(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, persist=True, verbose=False)
        result = results[0]

        pred_img = frame.copy()

        # Draw bounding boxes and labels for detected objects
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw rectangle around detected object
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Put label text above the bounding box
                cv2.putText(pred_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection", pred_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

# Handle command-line arguments
parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
parser.add_argument("--mode", choices=["image", "video", "webcam"], required=True, help="Inference mode")
parser.add_argument("--input", type=str, help="Input file path for image or video")
parser.add_argument("--output", type=str, help="Output file path for image or video")

args = parser.parse_args()

if args.mode == "image":
    if not args.input or not args.output:
        print("For image mode, --input and --output must be specified")
    inference_image(args.input, args.output)

elif args.mode == "video":
    if not args.input or not args.output:
        print("For video mode, --input and --output must be specified")
    inference_video(args.input, args.output)

elif args.mode == "webcam":
    inference_webcam()