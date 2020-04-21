from PIL import Image
import numpy as np 
import os

from sampler_kitti import crop_for_perfect_scaling
import sys, os
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, "../../"))
from c3d.utils.cam import scale_image

def crop_np(image, x_start, y_start, x_size, y_size):
    x_start = int(x_start)
    y_start = int(y_start)
    x_size = int(x_size)
    y_size = int(y_size)
    image_cropped = image[y_start:y_start+y_size, x_start:x_start+x_size ]
    return image_cropped

### 0. settings
scale = 2
x_crop_size = 40 #704
y_crop_size = 40 #352
x_crop_start = 0
y_crop_start = 0
x_half_crop_size = x_crop_size/2
y_half_crop_size = y_crop_size/2
x_half_crop_start = x_crop_start/2
y_half_crop_start = y_crop_start/2
image_path = 'vis_intr/0_0_side_rgb_0.png'
save_folder = 'vis_2'


### 1. prepare the initial image
# image = Image.open(image_path)
# image_np = np.array(image)

image_np = np.zeros((120, 120, 3)).astype(np.float32)
for i in range(120):
    for j in range(120):
        image_np[i,j] = i+j

x_start_size, y_start_size = crop_for_perfect_scaling(image_np, 2, h_dim=0, w_dim=1)
image_np = image_np[:y_start_size, :x_start_size]
image = Image.fromarray(np.round(image_np).astype(np.uint8))

### 2. scale -> crop
new_width = int(x_start_size / scale)
new_height = int(y_start_size / scale)

image_scaled = scale_image(image_np, new_width, new_height, torch_mode=True, nearest=False, raw_float=False, align_corner=False)
# image_scaled = scale_image(image, new_width, new_height, torch_mode=False, nearest=False, raw_float=False, align_corner=False)
# image_scaled = image.resize( (new_width, new_height), Image.BILINEAR)
image_scaled_np = np.array(image_scaled)

image_scaled_cropped_np = crop_np(image_scaled_np, x_half_crop_start, y_half_crop_start, x_half_crop_size, y_half_crop_size)
image_scaled_cropped = Image.fromarray(np.round(image_scaled_cropped_np).astype(np.uint8))

### 3. crop -> scale
image_cropped_np = crop_np(image_np, x_crop_start, y_crop_start, x_crop_size, y_crop_size)
image_cropped = Image.fromarray(np.round(image_cropped_np).astype(np.uint8))

new_width = int(x_crop_size / scale)
new_height = int(y_crop_size / scale)
image_cropped_scaled_np = scale_image(image_cropped_np, new_width, new_height, torch_mode=True, nearest=False, raw_float=False, align_corner=False)
# image_cropped_scaled = scale_image(image_cropped, new_width, new_height, torch_mode=False, nearest=False, raw_float=False, align_corner=False)
# image_cropped_scaled = image_cropped.resize( (new_width, new_height), Image.BILINEAR)
# image_cropped_scaled_np = np.array(image_cropped_scaled)
image_cropped_scaled = Image.fromarray(np.round(image_cropped_scaled_np).astype(np.uint8))

print(np.abs(image_cropped_scaled_np.astype(np.float32) - image_scaled_cropped_np.astype(np.float32)).max())

### 4. visualization
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

image.save(os.path.join(save_folder, 'raw.png'))
image_scaled_cropped.save(os.path.join(save_folder, 'scale_crop.png'))
image_cropped_scaled.save(os.path.join(save_folder, 'crop_scale.png'))


# ### toy example showing whether PIL.Image.resize use align_corner or not
# image_4 = np.array([[0,3,6,9]]).astype(np.float32)
# image_4_pil = Image.fromarray(image_4)
# image_2_pil = image_4_pil.resize((2,1), resample=Image.BILINEAR)
# image_2 = np.array(image_2_pil)
# print(image_2)


# image_4 = np.array([[10, 20, 0,2,4,6, 10, 20]]).astype(np.float32)
# image_4_pil = Image.fromarray(image_4)
# image_2_pil = image_4_pil.resize((4,1), resample=Image.BILINEAR)
# image_2 = np.array(image_2_pil)
# print(image_2)