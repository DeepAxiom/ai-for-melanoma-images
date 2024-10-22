import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import csv

class DeviceManager:
    """Manages the device configuration for computations."""
    @staticmethod
    def get_device():
        device = torch.device("cuda")
        print(f"Using device: {device}")
        return device

class ImageProcessor:
    """Handles loading and processing images."""
    @staticmethod
    def load_image(image_path):
        """Loads an image from the specified path."""
        image = Image.open(image_path)
        return np.array(image.convert("RGB"))

    @staticmethod
    def resize_image(image, size=(100, 100)):
        """Resizes the image to the specified size."""
        return cv2.resize(image, size)

class MaskHandler:
    """Generates and manages masks for images."""
    def __init__(self, model, image, output_dir, mask_color=(0, 0, 255)):
        self.model = model
        self.image = image
        self.output_dir = output_dir
        self.mask_color = mask_color

    def show_and_save_mask(self, mask, output_prefix):
        """Displays and saves the mask overlayed on the original image."""
        img_with_mask = self.image.copy()
        alpha_channel = np.ones((img_with_mask.shape[0], img_with_mask.shape[1]), dtype=np.uint8) * 255  
        img_with_mask[mask] = (img_with_mask[mask] * 0.5 + np.array(self.mask_color) * 0.5).astype(np.uint8)  

        img_with_mask_rgba = np.dstack((img_with_mask, alpha_channel))
        mask_image_path = os.path.join(self.output_dir, f"{output_prefix}_with_mask.png")
        plt.figure(figsize=(20, 20))
        plt.imshow(img_with_mask_rgba)
        plt.axis('off')
        plt.savefig(mask_image_path, bbox_inches='tight', format='png')
        plt.close()

        masked_area = np.zeros_like(self.image, dtype=np.uint8)  
        alpha_channel_transparent = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8) 
        masked_area[mask] = self.image[mask]  
        alpha_channel_transparent[mask] = 255  
        masked_area_rgba = np.dstack((masked_area, alpha_channel_transparent))
        cropped_image_path = os.path.join(self.output_dir, f"{output_prefix}_cropped_mask.png")
        cropped_image = Image.fromarray(masked_area_rgba, mode='RGBA')
        cropped_image.save(cropped_image_path)  
        
        return cropped_image_path, mask_image_path

class ColorAnalyzer:
    """Analyzes colors in images and calculates various metrics."""
    def __init__(self, image_path, n_clusters=5, threshold_black=(0, 0, 0)):
        self.image_path = image_path
        self.n_clusters = n_clusters
        self.threshold_black = threshold_black
        self.filtered_colors = []
        self.recalculated_percentages = []
        self.area = 0
        self.perimeter = 0
        self.diameter = 0
        self.asymmetry = 0
        self.irregularity = 0
        self.binary_mask = None  

    def load_and_process_image(self):
        """Loads and processes the image for color extraction."""
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Cannot load the image: {self.image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return ImageProcessor.resize_image(image_rgb)

    def find_dominant_colors(self, image):
        """Finds dominant colors in the image using K-Means clustering."""
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)       
        kmeans.fit(pixels)

        dominant_colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100

        self._filter_black_colors(dominant_colors, percentages)

    def _filter_black_colors(self, dominant_colors, percentages):
        """Filters out black colors and recalculates percentages."""
        for i, color in enumerate(dominant_colors):
            if not np.array_equal(color, self.threshold_black):  
                self.filtered_colors.append(color)
                self.recalculated_percentages.append(percentages[i])

        total_pixels_filtered = sum(self.recalculated_percentages)
        if total_pixels_filtered > 0:
            self.recalculated_percentages = [(p / total_pixels_filtered) * 100 for p in self.recalculated_percentages]

    def calculate_area_perimeter(self):
        """Calculates area and perimeter of the mask."""
        original_image = cv2.imread(self.image_path)
        grayscale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, self.binary_mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            self.area = cv2.contourArea(largest_contour)
            self.perimeter = cv2.arcLength(largest_contour, True)

            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            self.diameter = 2 * radius

    def calculate_asymmetry(self):
        """Calculates asymmetry using Intersection over Union (IoU) metric."""
        grayscale = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)

        h, w = binary_mask.shape
        left_half = binary_mask[:, :w//2]
        right_half = binary_mask[:, w//2:]

        right_half_flipped = np.fliplr(right_half)

        if left_half.shape[1] != right_half_flipped.shape[1]:
            right_half_flipped = np.pad(right_half_flipped, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        intersection = np.logical_and(left_half, right_half_flipped)
        union = np.logical_or(left_half, right_half_flipped)

        self.asymmetry = 1 - (np.sum(intersection) / np.sum(union)) if np.sum(union) != 0 else 0

    def calculate_border_irregularity(self):
        """Calculates border irregularity by comparing actual perimeter with ideal perimeter."""
        if self.perimeter > 0 and self.area > 0:
            expected_perimeter = 2 * np.sqrt(np.pi * self.area)
            self.irregularity = self.perimeter / expected_perimeter

    def calculate_simpson_index(self):
        """Calculates Simpson's index."""
        proportions = [p / sum(self.recalculated_percentages) for p in self.recalculated_percentages]
        return 1 - sum([p ** 2 for p in proportions])

    def calculate_standard_deviation(self):
        """Calculates the standard deviation of the percentages."""
        return np.std(self.recalculated_percentages)

    def display_and_save_colors(self, output_image):
        """Saves the filtered colors and mask to a PNG file."""
        sorted_indices = np.argsort(self.recalculated_percentages)[::-1]
        sorted_colors = [self.filtered_colors[i] for i in sorted_indices]
        sorted_percentages = [self.recalculated_percentages[i] for i in sorted_indices]

        fig, axes = plt.subplots(1, len(sorted_colors) + 1, figsize=(12, 6))

        axes[0].imshow(self.binary_mask, cmap='gray')
        axes[0].set_title("Binary Mask")
        axes[0].axis('off')

        for i, color in enumerate(sorted_colors):
            axes[i + 1].imshow([[color]])
            axes[i + 1].axis('off')
            axes[i + 1].set_title(f"{sorted_percentages[i]:.2f}%")

        plt.tight_layout()
        plt.savefig(output_image)
        plt.close()
        print(f"Colors and mask saved in {output_image}")

    def save_colors_to_csv(self, csv_file, image_paths):
        """Saves colors and their percentages to a CSV file."""
        simpson_index = self.calculate_simpson_index()
        std_dev = self.calculate_standard_deviation()

        sorted_indices = np.argsort(self.recalculated_percentages)[::-1]
        sorted_colors = [self.filtered_colors[i] for i in sorted_indices]
        sorted_percentages = [self.recalculated_percentages[i] for i in sorted_indices]

        file_exists = os.path.isfile(csv_file)

        self.calculate_asymmetry()
        self.calculate_border_irregularity()

        # Extraer el nombre del archivo original desde la ruta
        original_image_name = os.path.basename(self.image_path)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                headers = ['Image', 'Cropped Mask Path', 'Mask Path', 
                           'Dominant Color', 'Percentage', 
                           'Area', 'Perimeter', 'Diameter', 
                           'Asymmetry', 'Irregularity', 
                           'Simpson Index', 'Standard Deviation']
                writer.writerow(headers)

            # Agregar el nombre del archivo original a los datos
            data = [original_image_name, os.path.relpath(self.image_path), 
                    os.path.relpath(image_paths[0]), 
                    os.path.relpath(image_paths[1])]  # Use relative paths for images

            for i, color in enumerate(sorted_colors):
                color_str = f"{color[0]},{color[1]},{color[2]}"
                data.append(color_str)
                data.append(f"{sorted_percentages[i]:.2f}%")

            data.extend([
                f"{self.area:.2f}", 
                f"{self.perimeter:.2f}", 
                f"{self.diameter:.2f}",
                f"{self.asymmetry:.2f}", 
                f"{self.irregularity:.2f}",
                f"{simpson_index:.2f}", 
                f"{std_dev:.2f}"
            ])

            writer.writerow(data)

        print(f"Colors and percentages saved in {csv_file}")


class SAMModel:
    """Handles the SAM model loading and usage."""
    def __init__(self, model_cfg, checkpoint, device):
        self.device = device
        self.model = self.load_model(model_cfg, checkpoint)

    def load_model(self, model_cfg, checkpoint):
        """Loads the SAM model based on the configuration and checkpoint."""
        from sam2.build_sam import build_sam2
        return build_sam2(model_cfg, checkpoint, device=self.device, apply_postprocessing=False)

    def generate_masks(self, image):
        """Generates masks using the SAM model."""
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
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
        return mask_generator.generate(image)

def find_images_in_folder(folder_path):
    """Finds and returns a list of image paths within the folder and its subfolders."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')  # formatos de imagen comunes
    image_paths = []

    # Recorre la carpeta base y todas las subcarpetas
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    
    return image_paths

def main(folder_path='images_folder', checkpoint_path="../checkpoints/sam2.1_hiera_large.pt", 
         model_cfg_path="configs/sam2.1/sam2.1_hiera_l.yaml"):
    """Main function to execute the image processing and analysis pipeline for all images in a folder."""
    device = DeviceManager.get_device()
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    np.random.seed(3)

    image_paths = find_images_in_folder(folder_path)
    
    if not image_paths:
        print(f"No se encontraron imágenes en la carpeta: {folder_path}")
        return

    print(f"Se encontraron {len(image_paths)} imágenes en la carpeta {folder_path}.")
    
    sam_model = SAMModel(model_cfg_path, checkpoint_path, device)

    # Procesar cada imagen encontrada
    for image_path in image_paths:
        print(f"Procesando imagen: {image_path}")
        
        # Crear un directorio de salida basado en el nombre de la imagen
        image_base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(os.getcwd(), image_base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Cargar y procesar la imagen
        image = ImageProcessor.load_image(image_path)

        # Generar máscaras con SAM
        masks = sam_model.generate_masks(image)

        if masks:
            mask_handler = MaskHandler(sam_model, image, output_dir)
            largest_mask = max(masks, key=lambda x: x['area'])
            mask_array = largest_mask['segmentation']
            height, width = mask_array.shape
            
            is_skin_mask = np.any(mask_array[0, :]) or np.any(mask_array[height - 1, :]) or \
                           np.any(mask_array[:, 0]) or np.any(mask_array[:, width - 1])

            if is_skin_mask:
                print("The largest mask belongs to skin and will be ignored.")
                masks = [mask for mask in masks if mask['segmentation'].sum() > 0 and not (
                    np.any(mask['segmentation'][0, :]) or
                    np.any(mask['segmentation'][height - 1, :]) or
                    np.any(mask['segmentation'][:, 0]) or
                    np.any(mask['segmentation'][:, width - 1])
                )]

            if masks:
                largest_mask = max(masks, key=lambda x: x['area'])['segmentation']
                print("Selected mask area:", largest_mask.sum())
                cropped_mask_path, mask_image_path = mask_handler.show_and_save_mask(largest_mask, image_base_name)
            else:
                print("No valid masks found.")
                continue
        else:
            print("No masks were generated.")
            continue

        # Analizar los colores
        analyzer = ColorAnalyzer(cropped_mask_path)  
        
        try:
            processed_image = analyzer.load_and_process_image()
            analyzer.find_dominant_colors(processed_image)
            analyzer.calculate_area_perimeter()  
            output_image_path = os.path.join(output_dir, f"{image_base_name}_colors.png")
            analyzer.display_and_save_colors(output_image_path)  
            
            csv_file_path = os.path.join(os.getcwd(), "database.csv")
            analyzer.save_colors_to_csv(csv_file_path, [cropped_mask_path, mask_image_path])  
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    main(folder_path='images')