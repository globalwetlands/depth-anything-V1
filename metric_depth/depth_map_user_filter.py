import argparse
import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize

# Global settings
FY = 1109
FX = 1109
FINAL_HEIGHT = 1080
FINAL_WIDTH = 1920
NYU_DATA = True  # This uses FL as FY and FX
DATASET = "nyu"  # Let's not pick a fight with the model's dataloader
MODEL = "indoor"  # Change model here | Options: "indoor" or "outdoor"
CAMERA = "left"  # Change camera here | Options: "left" or "right" | This will adjust which camera focal length will be used


def load_data(file_path):
    """
    Load data from a CSV file and preprocess the columns.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed data as a DataFrame.
    """
    df = pd.read_csv(file_path)
    df["deployment-code"] = df["deployment-code"]
    df["calibration-id"] = df["calibration-id"].astype(str)
    df["camera"] = df["camera"].astype(str)
    df["frame"] = df["frame"].astype(int)
    df["user"] = df["user"].astype(str)
    df["sizes"] = df["sizes"].astype(str)
    return df


def filter_data(data, user="Jade"):
    """
    Filter data by user.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        user (str): User to filter by (default: "Jade").

    Returns:
        pd.DataFrame: Filtered data.
    """
    return data[data["user"] == user].copy()


def count_sequences(frames):
    """
    Count the number of sequences and return the start indices of each sequence.

    Args:
        frames (list): List of frame numbers.

    Returns:
        tuple: Total number of sequences and the start indices of each sequence.
    """
    sequences = 1
    sequence_indices = [0]  # Start of each sequence
    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1] and frames[i] != frames[i - 1] + 1:
            sequences += 1
            sequence_indices.append(i)
    return sequences, sequence_indices


def process_images(model, filtered_df, output_csv_file):
    """
    Processes images using the specified depth estimation model. It reads the focal length from the filtered DataFrame,
    applies the model to each image, and saves the depth maps images to the output directory.
    It will also record calibration-id, deployment-code, frame, focal_length, z-min, and z-max for each image onto a new CSV file.

    Args:
        model: The depth estimation model.
        filtered_df (pd.DataFrame): Filtered DataFrame containing focal length data.
        output_csv_file (str): Path to the new CSV file where processed data will be saved.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Create new columns safely
    filtered_df[f"z-min-{MODEL}-{CAMERA}"] = np.nan
    filtered_df[f"z-max-{MODEL}-{CAMERA}"] = np.nan

    frames = filtered_df["frame"].sort_values().tolist()
    sequences, sequence_indices = count_sequences(frames)

    for i in tqdm(range(len(frames)), desc="Processing Images"):
        current_frame = frames[i]
        focal_length_row = filtered_df[filtered_df["frame"] == current_frame].iloc[0]

        deployment_code = focal_length_row["deployment-code"]
        calibration_id = focal_length_row["calibration-id"]
        frame = focal_length_row["frame"]

        FL = focal_length_row[f"focal-length-{CAMERA}"]
        cy_left = focal_length_row[f"cy-{CAMERA}"]
        cx_left = focal_length_row[f"cx-{CAMERA}"]

        if pd.isna(cy_left) or pd.isna(cx_left):
            print(
                f"Skipping frame {frame} due to NaN values in cy-{CAMERA} or cx-{CAMERA}"
            )
            continue

        print(
            f"Processing {deployment_code}_{calibration_id}_{CAMERA}_{frame} | FL: {FL} | cy-left: {cy_left} | cx-left: {cx_left}"
        )

        image_path = os.path.join(
            INPUT_DIR, f"{deployment_code}_{calibration_id}_L_{frame}.jpg"
        )
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping.")
            continue

        color_image = Image.open(image_path).convert("RGB")
        original_width, original_height = color_image.size
        image_tensor = (
            transforms.ToTensor()(color_image)
            .unsqueeze(0)
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )

        pred = model(image_tensor, dataset=DATASET)
        if isinstance(pred, dict):
            pred = pred.get("metric_depth", pred.get("out"))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]

        predm = pred.squeeze().detach().cpu().numpy()
        resized_color_image = color_image.resize(
            (FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS
        )
        resized_pred = Image.fromarray(predm).resize(
            (FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST
        )

        focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
        x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
        x = (x - cx_left) / focal_length_x
        y = (y - cy_left) / focal_length_y
        z = np.array(resized_pred)

        z_min = z.min()
        z_max = z.max()

        filtered_df.loc[
            filtered_df["frame"] == current_frame,
            [f"z-min-{MODEL}-{CAMERA}", f"z-max-{MODEL}-{CAMERA}"],
        ] = [z_min, z_max]

        print(f"z-min-{MODEL}: {z_min} | z-max-{MODEL}: {z_max}")

        z_norm = (z - z_min) / (z_max - z_min)
        imgdepth = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
        o3d.io.write_image(
            os.path.join(
                OUTPUT_DIR,
                os.path.splitext(os.path.basename(image_path))[0] + "_pred04.png",
            ),
            imgdepth,
        )

    # Save the updated DataFrame to a new CSV file
    filtered_df.to_csv(output_csv_file, index=False)


def main(model_name, pretrained_resource, focal_length_file, input_dir, output_dir):
    """
    Main function that sets up the model and processes images using the specified parameters.

    Args:
        model_name (str): Name of the model to test.
        pretrained_resource (str): Pretrained resource to use for fetching weights.
        focal_length_file (str): CSV file containing the focal length.
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save output images.
    """
    global INPUT_DIR, OUTPUT_DIR

    # Read and filter data from CSV
    df = load_data(focal_length_file)
    filtered_df = filter_data(df, user="Jade", size="sizes00")

    INPUT_DIR = input_dir  # Image input directory
    OUTPUT_DIR = output_dir  # Images output directory

    # Define the output CSV file name
    output_csv_file = focal_length_file.replace(".csv", "_processed_Jade_sizes00.csv")

    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource.format(MODEL=MODEL)
    model = build_model(config).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    process_images(model, filtered_df, output_csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="zoedepth", help="Name of the model to test"
    )
    parser.add_argument(
        "-p",
        "--pretrained_resource",
        type=str,
        default=f"local::./checkpoints/depth_anything_metric_depth_{MODEL}.pt",  # Default model is indoor
        help="Pretrained resource to use for fetching weights.",
    )
    parser.add_argument(
        "-f",
        "--focal_length_file",
        type=str,
        required=True,
        help="CSV file containing the focal length.",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="./my_test/input",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./my_test/output",
        help="Directory to save output images.",
    )

    args = parser.parse_args()
    main(
        args.model,
        args.pretrained_resource,
        args.focal_length_file,
        args.input_dir,
        args.output_dir,
    )
