"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-preliminary-docker-evaluation | gzip -c > example-algorithm-preliminary-docker-evaluation.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

from pathlib import Path
import glob
import SimpleITK
import torch
import json
import os
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import zoom
# from eval_utils import compute_absolute_volume_difference, compute_dice_f1_instance_difference  # Import the evaluation functions

import numpy as np
import warnings

def run():
    INPUT_PATH = Path("/input")
    OUTPUT_PATH = Path("/output")
    
    # Load image files and preprocess
    non_contrast_ct = load_nii(f"{INPUT_PATH}/images/non-contrast-ct", is_ncct=True)
    ct_angiography = load_nii(f"{INPUT_PATH}/images/preprocessed-ct-angiography", is_cta=True)
    perfusion_ct = load_nii(f"{INPUT_PATH}/images/perfusion-ct", is_ctp=True)
    mtt = load_nii(f"{INPUT_PATH}/images/preprocessed-mtt-map", is_t=True)
    tmax = load_nii(f"{INPUT_PATH}/images/preprocessed-tmax-map", is_t=True)
    
    # Stack the image modalities
    img_stack = torch.cat((non_contrast_ct, ct_angiography, perfusion_ct, mtt, tmax), dim=1)

    # Load your model
    # Instantiate VisionTransformerForSegmentation
    vit_args = dataclasses.asdict(VisionTransformerArgs(patch_size=(8, 8, 8)))
    model = VisionTransformerForSegmentation(**vit_args)

    # Load the saved state dictionary
    checkpoint = torch.load(f"{INPUT_PATH}/yes_dwi_4inputs_1.pth",map_location ='cpu')
    # Load model and optimizer state dictionaries
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # model = torch.load(f"{INPUT_PATH}/model1.pth",map_location ='cpu')
    # model.eval()

    # Predict infarct
    with torch.no_grad():
        stroke_lesion_segmentation = model(img_stack)
        stroke_lesion_segmentation = (stroke_lesion_segmentation > 0.5).float().cpu().numpy()

    # # Load ground truth for evaluation
    # ground_truth = load_nii(INPUT_PATH / "images/ground-truth-lesion-mask.nii")  # Adjust path accordingly

    # # Compute evaluation metrics
    # voxel_size = np.prod(ground_truth.header.get_zooms())  # Assuming voxel size is retrieved like this
    # abs_vol_diff = compute_absolute_volume_difference(ground_truth, stroke_lesion_segmentation, voxel_size)
    # f1_score, instance_count_diff, dice_score = compute_dice_f1_instance_difference(ground_truth, stroke_lesion_segmentation)

    # # Print or log the evaluation metrics
    # print(f"Absolute Volume Difference: {abs_vol_diff:.2f} ml")
    # print(f"Dice Score: {dice_score:.2f}")
    # print(f"Instance Count Difference: {instance_count_diff}")
    # print(f"F1 Score: {f1_score:.2f}")

    # Save output
    write_array_as_image_file(
        location=f"{OUTPUT_PATH}/images/stroke-lesion-segmentation",
        array=stroke_lesion_segmentation,
    )

    return 0

def load_nii(file_path, target_shape=(64, 64, 64), is_ncct=False, is_cta=False, is_ctp=False, is_cb=False, is_t=False, window_center=40, window_width=80):
    input_files = glob.glob(os.path.join(file_path, "*.mha"))
    mha_file = input_files[0]
    result = SimpleITK.ReadImage(mha_file)
    img = SimpleITK.GetArrayFromImage(result)
    img = img.astype(np.uint8)

    if is_ncct:
        windowed_volume = apply_window(img, window_center, window_width)
        skull_stripped_volume = skull_strip(windowed_volume)
        img = windowed_volume - skull_stripped_volume

    if is_cta:
        img = apply_window(img, window_center, window_width)

    if is_ctp:
        if img.ndim == 4:
            img = img[:,:,:,0]
            windowed_volume = apply_window(img, window_center, window_width)
            img = windowed_volume

    if is_cb:
        img = apply_window(img, window_center, window_width)

    if is_t:
        img = apply_window(img, window_center, window_width)
    
    
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)

    img = torch.tensor(img, dtype=torch.float32)
    img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=target_shape, mode='trilinear', align_corners=True)
    return img

def apply_window(image, window_center, window_width):
    """
    Applies windowing to an image based on the given center and width.

    :param image: The input image.
    :param window_center: Center of the windowing range.
    :param window_width: Width of the windowing range.
    :return: Windowed image.
    """
    min_window = window_center - window_width // 2
    max_window = window_center + window_width // 2
    windowed_image = np.clip((image - min_window) / (max_window - min_window) * 255.0, 0, 255)
    return windowed_image.astype(np.uint8)

def skull_strip(image):
    """
    Performs skull stripping on the given image using thresholding and morphological operations.

    :param image: The input image.
    :return: Skull-stripped image.
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, binary_dilation, ball, remove_small_objects
    from scipy.ndimage import binary_fill_holes

    threshold = threshold_otsu(image)  # Calculate Otsu's threshold
    binary_image = image > threshold  # Create a binary mask
    footprint = ball(2)
    cleaned_image = binary_closing(binary_image, footprint=footprint)  # Close small holes
    cleaned_image = binary_fill_holes(cleaned_image)  # Fill holes
    cleaned_image = remove_small_objects(cleaned_image, min_size=2000)  # Remove small objects
    brain_mask = binary_dilation(cleaned_image, footprint=footprint)  # Dilate the mask
    skull_stripped_image = image * brain_mask  # Apply the mask to the image
    return skull_stripped_image

def write_array_as_image_file(*, location, array):
    """
    Saves a numpy array as an image file using SimpleITK.

    :param location: Directory to save the image.
    :param array: The numpy array to save as an image.
    """
    # Convert location to a Path object
    location_path = Path(location)
    
    # Create the directory if it doesn't exist
    location_path.mkdir(parents=True, exist_ok=True)
    
    suffix = ".mha"
    image = SimpleITK.GetImageFromArray(array)
    
    # Save the image
    SimpleITK.WriteImage(
        image,
        str(location_path / f"output{suffix}"),
        useCompression=True,
    )

# Model classes
import torch
import torch.nn as nn
import dataclasses
#!pip install --user unfoldNd
import unfoldNd

# Image to Patches for 3D Input
class ImageToPatches3D(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = unfoldNd.UnfoldNd(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        assert len(x.size()) == 5  # Ensure 5D input tensor (batch_size, channels, height, width, depth)
        x_unfolded = self.unfold(x)
        x_unfolded = x_unfolded.permute(0, 2, 1)
        return x_unfolded
    
# Vision Transformer for Segmentation with 3D Input
class VisionTransformerForSegmentation(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, out_channels, embed_size, num_blocks, num_heads, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout

        heads = [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        self.transformer_layers = nn.Sequential(
            nn.BatchNorm3d(num_features=in_channels),
            VisionTransformerInput(image_size, patch_size, in_channels, embed_size),
            nn.Sequential(*heads),
            OutputProjection(image_size, patch_size, embed_size, out_channels)
        )

        self.unet = UNet3D(in_channels, out_channels)

        self.activation = nn.Sigmoid() 

    def forward(self, x):
        x_transformer = self.transformer_layers(x.float())
        x_unet = self.unet(x.float())
        x_combined = torch.cat((x_transformer, x_unet), dim=1)
        x_mean = torch.mean(x_combined, dim=1, keepdim=True)
        return self.activation(x_mean)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        
        self.decoder4 = self.upconv_block(128, out_channels)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))

        dec4 = self.decoder4(F.interpolate(enc2, scale_factor=1, mode='trilinear', align_corners=False))
        return dec4

# Helper modules

# Vision Transformer Input Layer
class VisionTransformerInput(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        super().__init__()
        self.i2p3d = ImageToPatches3D(image_size, patch_size)
        self.pe = PatchEmbedding3D(patch_size[0] * patch_size[1] * patch_size[2] * in_channels, embed_size)
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) * (image_size[2] // patch_size[2])
        self.position_embed = nn.Parameter(torch.randn(1, num_patches, embed_size))

    def forward(self, x):
        x = self.i2p3d(x)
        x = self.pe(x)
        x = x + self.position_embed
        return x

# Patch Embedding for 3D Input
class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.embed_layer = nn.Linear(in_features=in_channels, out_features=embed_size)

    def forward(self, x):
        assert len(x.size()) == 3
        B, T, C = x.size()
        x = self.embed_layer(x)
        return x

# Multi-Layer Perceptron
class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.layers(x)

# Self-Attention Encoder Block
class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.ln1 = nn.LayerNorm(embed_size)
        self.mha = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MultiLayerPerceptron(embed_size, dropout)

    def forward(self, x):
        y = self.ln1(x)
        x = x + self.mha(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

# Output Projection
class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.projection = nn.Linear(embed_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels)
        self.fold = unfoldNd.FoldNd(output_size=image_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, C = x.shape
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = self.fold(x)
        return x

# Example usage
@dataclasses.dataclass
class VisionTransformerArgs:
    """Arguments to the VisionTransformerForSegmentation."""
    image_size: tuple = (64, 64, 64)
    patch_size: tuple = (8, 8, 8)
    in_channels: int = 5
    out_channels: int = 1
    embed_size: int = 128
    num_blocks: int = 32
    num_heads: int = 16
    dropout: float = 0.5

# def load_image_file_as_array(*, location):
#     # Use SimpleITK to read a file
#     input_files = glob(str(location / "*.mha"))
#     result = SimpleITK.ReadImage(input_files[0])

#     # Convert it to a Numpy array
#     return SimpleITK.GetArrayFromImage(result)


# def write_array_as_image_file(*, location, array):
#     location.mkdir(parents=True, exist_ok=True)

#     suffix = ".mha"
#     print(str(location / f"output{suffix}"))
#     image = SimpleITK.GetImageFromArray(array)
#     print(sum(image))
#     SimpleITK.WriteImage(
#         image,
#         location / f"output{suffix}",
#         useCompression=True,
#     )


# def predict_infarct(preprocessed_tmax, cutoff=9):
#     ''' We are creating a simple lesion prediction based on the introductory Git-Repository example:
#     https://github.com/ezequieldlrosa/isles24
#     In this exapmle, we load the preprocessed Tmax map and threshold it at a cutoff of 9 (s).'''

#     ################################################################################################################
#     #################################### Beginning of your prediction method. ######################################
#     # todo replace with your best model here!

#     prediction = preprocessed_tmax > cutoff
#     ################################################################################################################

#     return prediction.astype(int)


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)
if __name__ == "__main__":
    raise SystemExit(run())
