import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2  
device = torch.device("cuda")
print(f"using device: {device}")

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)

def show_and_save_mask(image, mask, mask_color=(0, 0, 255)):
    img_with_mask = image.copy()
    
    alpha_channel = np.ones((img_with_mask.shape[0], img_with_mask.shape[1]), dtype=np.uint8) * 255  
    img_with_mask[mask] = (img_with_mask[mask] * 0.5 + np.array(mask_color) * 0.5).astype(np.uint8)  

    img_with_mask_rgba = np.dstack((img_with_mask, alpha_channel))

    plt.figure(figsize=(20, 20))
    plt.imshow(img_with_mask_rgba)
    plt.axis('off')
    plt.savefig("image_with_mask.png", bbox_inches='tight', format='png') 
    plt.show()

    masked_area = np.zeros_like(image, dtype=np.uint8)  
    alpha_channel_transparent = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) 

    masked_area[mask] = image[mask]  
    alpha_channel_transparent[mask] = 255  

    masked_area_rgba = np.dstack((masked_area, alpha_channel_transparent))

    cropped_image = Image.fromarray(masked_area_rgba, mode='RGBA')
    cropped_image.save("cropped_mask.png")  


image = Image.open('images/2.jpg')
image = np.array(image.convert("RGB"))

plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')
plt.show()

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64, 
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.5,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
)

masks = mask_generator_2.generate(image)

if masks:
    largest_mask = max(masks, key=lambda x: x['area'])
    
    mask_array = largest_mask['segmentation']
    
    height, width = mask_array.shape
    
    is_skin_mask = np.any(mask_array[0, :]) or np.any(mask_array[height - 1, :]) or \
                   np.any(mask_array[:, 0]) or np.any(mask_array[:, width - 1])

    if is_skin_mask:
        print("La máscara más grande pertenece a la piel y será ignorada.")
        masks = [mask for mask in masks if mask['segmentation'].sum() > 0 and not (
            np.any(mask['segmentation'][0, :]) or
            np.any(mask['segmentation'][height - 1, :]) or
            np.any(mask['segmentation'][:, 0]) or
            np.any(mask['segmentation'][:, width - 1])
        )]

        if masks:
            largest_mask = max(masks, key=lambda x: x['area'])['segmentation']
            print("Área de la máscara seleccionada:", largest_mask.sum())
            show_and_save_mask(image, largest_mask)
        else:
            print("No se encontraron máscaras válidas.")
    else:
        print("Área de la máscara más grande:", largest_mask['area'])
        show_and_save_mask(image, largest_mask)

else:
    print("No se generaron máscaras.")



#extracción del color - Promedio del histograma  
 # COn pip install opencv-python 

 # Binarización de Otso  

 #Area
 #Perímetro



