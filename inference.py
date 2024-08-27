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
import dataclasses
import unfoldNd

def run():
    INPUT_PATH = Path("/input")
    OUTPUT_PATH = Path("/output")
    
    # Load image files and preprocess
    non_contrast_ct = load_nii(f"{INPUT_PATH}/images/non-contrast-ct", is_ncct=True)
    ct_angiography = load_nii(f"{INPUT_PATH}/images/preprocessed-CT-angiography", is_cta=True)
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
    checkpoint = torch.load(f"{INPUT_PATH}/yes_dwi_4inputs_1.pth")
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Predict infarct
    with torch.no_grad():
        stroke_lesion_segmentation = model(img_stack)
        stroke_lesion_segmentation = (stroke_lesion_segmentation > 0.5).float().cpu().numpy()

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
    min_window = window_center - window_width // 2
    max_window = window_center + window_width // 2
    windowed_image = np.clip((image - min_window) / (max_window - min_window) * 255.0, 0, 255)
    return windowed_image.astype(np.uint8)

def skull_strip(image):
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, binary_dilation, ball, remove_small_objects
    from scipy.ndimage import binary_fill_holes

    threshold = threshold_otsu(image)
    binary_image = image > threshold
    footprint = ball(2)
    cleaned_image = binary_closing(binary_image, footprint=footprint)
    cleaned_image = binary_fill_holes(cleaned_image)
    cleaned_image = remove_small_objects(cleaned_image, min_size=2000)
    brain_mask = binary_dilation(cleaned_image, footprint=footprint)
    skull_stripped_image = image * brain_mask
    return skull_stripped_image

def write_array_as_image_file(*, location, array):
    location_path = Path(location)
    location_path.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"
    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        str(location_path / f"output{suffix}"),
        useCompression=True,
    )

class ImageToPatches3D(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = unfoldNd.UnfoldNd(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        assert len(x.size()) == 5
        x_unfolded = self.unfold(x)
        x_unfolded = x_unfolded.permute(0, 2, 1)
        return x_unfolded

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

class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.embed_layer = nn.Linear(in_channels, embed_size)

    def forward(self, x):
        B, T, C = x.size()
        x = self.embed_layer(x)
        return x

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

class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.mha = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MultiLayerPerceptron(embed_size, dropout)

    def forward(self, x):
        y = self.ln1(x)
        x = x + self.mha(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, out_channels):
        super().__init__()
        self.projection = nn.Linear(embed_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels)
        self.fold = unfoldNd.FoldNd(output_size=image_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, C = x.shape
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = self.fold(x)
        return x

@dataclasses.dataclass
class VisionTransformerArgs:
    image_size: tuple = (64, 64, 64)
    patch_size: tuple = (8, 8, 8)
    in_channels: int = 5
    out_channels: int = 1
    embed_size: int = 128
    num_blocks: int = 32
    num_heads: int = 16
    dropout: float = 0.5

if __name__ == "__main__":
    raise SystemExit(run())
