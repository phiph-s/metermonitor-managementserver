import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18
from torch import nn
from torchvision.transforms import transforms
from ultralytics import YOLO
import torchvision.transforms.functional as F

class MeterPredictor:
    """
    A class to perform water meter digit detection (using YOLO with OBB)
    and digit classification (using a TFLite model), for one image at a time.
    """

    def __init__(self, yolo_model_path, digit_classifier_model_path):
        """
        Initializes the YOLO model and TFLite Interpreter, so they can
        be reused for multiple images without re-initializing.

        Args:
            yolo_model_path (str): Path to the YOLO .pt file
            tflite_model_path (str): Path to the digit classification TFLite model
        """
        # Load YOLO model (oriented bounding box capable)
        self.model = YOLO(yolo_model_path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the model architecture (same as during training)
        self.digitmodel = resnet18(pretrained=False)  # Ensure pretrained=False if loading your weights
        self.digitmodel.fc = nn.Linear(self.digitmodel.fc.in_features, 11)  # Adjust for your classes

        # Load the saved weights
        state_dict = torch.load(digit_classifier_model_path, map_location=torch.device("cpu"))
        self.digitmodel.load_state_dict(state_dict)

        # Set the model to evaluation mode
        self.digitmodel.eval()

        # Move the model to the appropriate device (CPU or GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.digitmodel.to(self.device)

    def predict_single_image(self, input_image, output_folder=None, segments=7, contrast=1.0, padding=0.0, enhance_sharpness=False, extended_last_digit=False, shrink_last_3=False):
        """
        Predicts the water meter reading on a single image:
          - Runs YOLO detection for oriented bounding box (OBB)
          - Applies perspective transform to 'straighten' the meter
          - Splits the meter into 7 vertical segments
          - Classifies each segment using the TFLite model
          - Saves two images:
               1) The original with bounding box & predicted number (prefixed "pred_")
               2) The cropped/straightened region (prefixed "pred_cut_")

        Args:
            input_image_path (str): Path to the input image
            output_folder (str, None): Folder where to save results
        """
        # Ensure the output folder exists
        if output_folder: os.makedirs(output_folder, exist_ok=True)

        if type(input_image )is str:
            input_image_path = input_image
            image_name = os.path.basename(input_image_path)

            # 1. Run YOLO to predict OBB
            results = self.model.predict(input_image_path, save=False)
            obb_data = results[0].obb
        else:
            # input_image is a PIL Image
            results = self.model.predict(input_image, save=False)
            obb_data = results[0].obb

        # If no OBB found, bail out
        if (
                obb_data is None or
                obb_data.xyxyxyxy is None or
                len(obb_data.xyxyxyxy) == 0
        ):
            print(f"[INFO] No instances detected in the image {image_name}.")
            return

        # We'll use the first detected bounding box
        obb_coords = obb_data.xyxyxyxy[0].cpu().numpy()

        if type(input_image) is str:
            # 2. Load the image with OpenCV
            img = cv2.imread(input_image_path)
            if img is None:
                print(f"[ERROR] Unable to load image: {input_image_path}")
                return
        else:
            # input_image is a PIL Image
            img = np.array(input_image)

        if enhance_sharpness:
            img = cv2.addWeighted(img, 4, cv2.blur(img, (30, 30)), -4, 128)

        # Apply contrast adjustment
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

        # 3. Add padding at the bottom for annotation
        img = cv2.copyMakeBorder(img, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # 4. Reshape OBB coordinates into four (x,y) points
        points = obb_coords.reshape(4, 2).astype(np.float32)

        # Sort the points by y-coordinate (top to bottom)
        points = sorted(points, key=lambda x: x[1])

        # Separate top-left, top-right vs bottom-left, bottom-right
        if points[0][0] < points[1][0]:
            top_left, top_right = points[0], points[1]
        else:
            top_left, top_right = points[1], points[0]

        if points[2][0] < points[3][0]:
            bottom_left, bottom_right = points[2], points[3]
        else:
            bottom_left, bottom_right = points[3], points[2]

        # Reassemble into final order: [top-left, top-right, bottom-right, bottom-left]
        points = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

        # 5. Compute bounding box width/height
        width_a = np.linalg.norm(points[0] - points[1])
        width_b = np.linalg.norm(points[2] - points[3])
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(points[1] - points[2])
        height_b = np.linalg.norm(points[3] - points[0])
        max_height = max(int(height_a), int(height_b))

        # 6. Perspective transform to get the "front-facing" rectangle
        dst_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(points, dst_points)
        rotated_cropped_img = cv2.warpPerspective(img, M, (max_width, max_height))
        rotated_cropped_img_ext = None
        if extended_last_digit:
            rotated_cropped_img_ext = cv2.warpPerspective(img, M, (max_width, int(max_height * 1.2)))

        # 7. Split the cropped meter into segments (7) vertical parts for classification
        part_width = rotated_cropped_img.shape[1] // segments
        zw_values = []

        last_x = 0

        for i in range(segments):
            if shrink_last_3 and i >= segments - 3:
                t_part_width = int(part_width * 0.8)
            elif shrink_last_3:
                t_part_width = int(((part_width * segments) - (3 * part_width * 0.8)) / (segments - 3))
            else:
                t_part_width = part_width

            # Extract segment from last_x to last_x + t_part_width
            part = rotated_cropped_img[:, last_x: last_x + t_part_width]

            if extended_last_digit and i == segments - 1:
                part = rotated_cropped_img_ext[:, i * t_part_width: (i + 1) * t_part_width]

            # Adjust segment to 5:8 aspect ratio (width:height)
            part_height = part.shape[0]
            desired_width = (5 * part_height) // 8  # Calculate based on integer division
            current_width = part.shape[1]

            if current_width > desired_width:
                # Crop the center to desired width
                delta = current_width - desired_width
                left = delta // 2
                right = delta - left
                part = part[:, left:current_width - right]
            elif current_width < desired_width:
                # Pad left and right with white to reach desired width
                delta = desired_width - current_width
                left_pad = delta // 2
                right_pad = delta - left_pad
                part = cv2.copyMakeBorder(part, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            # Convert segment to PIL Image
            pil_part = Image.fromarray(cv2.cvtColor(part, cv2.COLOR_BGR2RGB))

            pipeline = [
                transforms.Resize((224, 224)),  # Resize to 224x224 (stretching if necessary)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            if padding > 0.0:
                pipeline.insert(1, transforms.CenterCrop(int(224 * (1 - padding))))
            preprocess = transforms.Compose(pipeline)

            tensor = preprocess(pil_part)
            input_tensor = tensor.unsqueeze(0).to(self.device)  # Add batch dimension

            # Perform inference
            with torch.no_grad():
                outputs = self.digitmodel(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Convert to probabilities
                # get top 2 values
                top3 = torch.topk(probabilities, 3)
                zw_values.append([(top3.indices[0].item(), top3.values[0].item()),
                                    (top3.indices[1].item(), top3.values[1].item()),
                                    (top3.indices[2].item(), top3.values[2].item())])

            # (Optional) draw bounding lines for each digit
            segment_x1 = last_x
            segment_x2 = last_x + t_part_width
            last_x = segment_x2
            cv2.rectangle(
                rotated_cropped_img,
                (segment_x1, 0),
                (segment_x2, rotated_cropped_img.shape[0]),
                (255, 0, 0),
                2
            )

            # add tensor to image for visualization
            simg = F.to_pil_image(tensor)
            # resize to original size of pil_part
            simg = simg.resize((part.shape[1], part.shape[0]))
            # add to the bottom of the image
            img[-simg.size[1]:, segment_x1:segment_x1+desired_width] = np.array(simg)

            # add red rectangle around the digit (-simg.size[1]:, segment_x1:segment_x2)
            cv2.rectangle(
                img,
                (segment_x1, img.shape[0] - simg.size[1]),
                (segment_x2, img.shape[0]),
                (0, 0, 255),
                2
            )

        if output_folder:

            # 8. Construct predicted number

            char_spacing = 30  # Adjust the spacing value as needed
            start_x = 10
            y = 30

            for val in zw_values:
                # get key with the highest value
                val = sorted(val, key=lambda x: x[1], reverse=True)[0]
                char = str(val[0]) if val[0] != 10 else "r"
                cv2.putText(img, char, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                start_x += char_spacing  # Increase the X position to simulate spacing

            start_x = 10
            y = 45

            for val in zw_values:
                # get key with the highest value
                val = sorted(val, key=lambda x: x[1], reverse=True)[0]
                char = str(round(val[1],1))
                cv2.putText(img, char, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                start_x += char_spacing  # Increase the X position to simulate spacing

            start_x = 10
            y = 70

            for val in zw_values:
                # get key with the highest value
                val = sorted(val, key=lambda x: x[1], reverse=True)[1]
                char = str(round(val[0],1)) if val[0] != 10 else "r"
                cv2.putText(img, char, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                start_x += char_spacing  # Increase the X position to simulate spacing

            start_x = 10
            y = 85

            for val in zw_values:
                # get key with the highest value
                val = sorted(val, key=lambda x: x[1], reverse=True)[1]
                char = str(round(val[1],1))
                cv2.putText(img, char, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                start_x += char_spacing  # Increase the X position to simulate spacing

            # Optional: draw bounding box for visualization
            for i in range(4):
                cv2.circle(
                    img,
                    (int(points[i][0]), int(points[i][1])),
                    5,
                    (0, 0, 255),
                    -1
                )
                cv2.line(
                    img,
                    (int(points[i][0]), int(points[i][1])),
                    (int(points[(i + 1) % 4][0]), int(points[(i + 1) % 4][1])),
                    (0, 0, 255),
                    2
                )

            # rotated_cropped_img_ext is only used if extended_last_digit is True
            # place to the right of the cropped meter
            if extended_last_digit:
                img[-rotated_cropped_img_ext.shape[0]:, max_width:max_width + rotated_cropped_img_ext.shape[1]] = rotated_cropped_img_ext

            # 11. Save the results
            pred_image_name = f"pred_{image_name}"
            pred_cut_image_name = f"pred_cut_{image_name}"

            pred_image_path = os.path.join(output_folder, pred_image_name)
            pred_cut_image_path = os.path.join(output_folder, pred_cut_image_name)

            cv2.imwrite(pred_image_path, img)
            cv2.imwrite(pred_cut_image_path, rotated_cropped_img)

            print(f"[INFO] Processed {image_name}. Saved results to:")
            print(f"       {pred_image_name}")
            print(f"       {pred_cut_image_name}")

        return zw_values

    def predict_folder(self, input_folder, output_folder=None, padding=0.0, segments=7, contrast=1.0, extended_last_digit=False, enhance_sharpness=False, shrink_last_3=False):
        """
        Predicts the water meter reading on all images in a folder.
        Saves the results as a CSV file with the format:
            filename, predicted_number, class1:confidence1 class2:confidence2 ...
        Args:
            input_folder (str): Path to the folder containing input images
        """
        results = {}

        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                input_image_path = os.path.join(input_folder, filename)
                fn_without_ext = os.path.splitext(filename)[0]
                results[fn_without_ext] = self.predict_single_image(
                    input_image_path=input_image_path,
                    output_folder=output_folder,
                    segments=segments,
                    contrast=contrast,
                    enhance_sharpness=enhance_sharpness,
                    extended_last_digit=extended_last_digit,
                    padding=padding,
                    shrink_last_3=shrink_last_3
                )
                print(results[fn_without_ext] )
                print(f"Predicted {filename} ({len(results)} / {len(os.listdir(input_folder))})")

        return results

    def save_to_csv(self, outputpath, results):
        # Save the results to a CSV file
        with open(outputpath, 'w') as f:
            for filename, values in results.items():
                if values is None:
                    print (f"Skipping {filename}")
                    continue
                f.write(f"{filename},")
                number_str = ""
                for val in values:

                    # get key with the highest value
                    h = None
                    for i in val:
                        if h is None or i[1] > h[1]:
                            h = i
                    number_str += str(h[0]) if h[0] != 10 else "r"
                print (f"{filename}: {number_str}")
                f.write(number_str + ",")
                for value in values:
                    f.write(" ".join([f"{i}:{v}" for i, v in value]) + ",")

                f.write("\n")