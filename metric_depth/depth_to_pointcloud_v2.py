import os
import glob
import torch
import numpy as np
from PIL import Image
import open3d as o3d
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize

# Constants
FL = 715.0873
FY = 1109
FX = 1109
NYU_DATA = False
FINAL_HEIGHT = 518
FINAL_WIDTH = 518
MODEL_NAME = "zoedepth"  # Modify this as needed
# PRETRAINED_RESOURCE = "local::./checkpoints/indoor/depth_anything_v2_metric_hypersim_vits.pth"  # Model path
PRETRAINED_RESOURCE = (
    "local::./checkpoints/depth_anything_metric_depth_outdoor.pt"  # Model path
)


def process_images(model, input_dir, output_dir):
    """
    Processes images and generate depth predictions,
    coloured images, and point clouds, saving the results to the output directory.

    Args:
        model: The depth prediction model.
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save output results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(
        os.path.join(input_dir, "*.jpg")
    )

    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            color_image = Image.open(image_path).convert("RGB")
            original_width, original_height = color_image.size
            FINAL_HEIGHT, FINAL_WIDTH = original_height, original_width

            image_tensor = (
                transforms.ToTensor()(color_image)
                .unsqueeze(0)
                .to("cuda" if torch.cuda.is_available() else "cpu")
            )

            pred = model(image_tensor, dataset="nyu")
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
            x = (x - FINAL_WIDTH / 2) / focal_length_x
            y = (y - FINAL_HEIGHT / 2) / focal_length_y
            z = np.array(resized_pred)

            Image.fromarray(predm).convert("L").resize(
                (FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST
            ).save(
                os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(image_path))[0] + "_pred01.png",
                )
            )

            p = colorize(pred.squeeze().detach().cpu().numpy(), cmap="magma_r")
            Image.fromarray(p).save(
                os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(image_path))[0] + "_pred02.png",
                )
            )

            pm = colorize(z, cmap="magma_r")
            Image.fromarray(pm).save(
                os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(image_path))[0] + "_pred03.png",
                )
            )

            z_norm = (z - z.min()) / (z.max() - z.min())
            imgdepth = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
            o3d.io.write_image(
                os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(image_path))[0] + "_pred04.png",
                ),
                imgdepth,
            )

            # points = np.stack(
            #     (np.multiply(x, z), np.multiply(y, z), z), axis=-1
            # ).reshape(-1, 3)
            # colors = np.array(resized_color_image).reshape(-1, 3) / 255.0

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud(
            #     os.path.join(
            #         output_dir,
            #         os.path.splitext(os.path.basename(image_path))[0] + ".ply",
            #     ),
            #     pcd,
            # )

        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Depth Maps")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./my_test/input",
        help="Path to the input directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./my_test/output",
        help="Path to the output directory",
    )

    args = parser.parse_args()

    config = get_config(MODEL_NAME, "eval", "nyu")
    config.pretrained_resource = PRETRAINED_RESOURCE
    model = build_model(config).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    process_images(model, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
