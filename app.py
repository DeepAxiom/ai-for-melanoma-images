import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv
import os
from scipy.stats import entropy

class ColorAnalyzer:
    def __init__(self, image_path, n_clusters=5, threshold_black=(0, 0, 0)):
        self.image_path = image_path
        self.n_clusters = n_clusters
        self.threshold_black = threshold_black
        self.colores_filtrados = []
        self.porcentajes_recalculados = []
        self.area = 0
        self.perimeter = 0
        self.diameter = 0
        self.asymmetry = 0
        self.irregularity = 0
        self.binary_mask = None  # Almacenar la máscara binaria

    def load_and_process_image(self):
        """Carga y procesa la imagen para la extracción de colores."""
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"No se puede cargar la imagen: {self.image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (100, 100))
        return image_resized

    def find_dominant_colors(self, image):
        """Encuentra los colores dominantes en la imagen utilizando K-Means."""
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)       
        kmeans.fit(pixels)

        colores_dominantes = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100

        # Filtrar colores para omitir los negros
        self._filter_black_colors(colores_dominantes, percentages)

    def _filter_black_colors(self, colores_dominantes, percentages):
        """Filtra los colores negros y recalcula los porcentajes."""
        for i, color in enumerate(colores_dominantes):
            if not np.array_equal(color, self.threshold_black):  
                self.colores_filtrados.append(color)
                self.porcentajes_recalculados.append(percentages[i])

        total_pixels_filtered = sum(self.porcentajes_recalculados)
        if total_pixels_filtered > 0:
            self.porcentajes_recalculados = [(p / total_pixels_filtered) * 100 for p in self.porcentajes_recalculados]

    def calculate_area_perimeter(self):
        """Calcula el área y el perímetro de la máscara."""
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
        """Calcula la asimetría utilizando la métrica de Intersección sobre Unión (IoU)."""
        original_image = cv2.imread(self.image_path)
        grayscale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)

        h, w = binary_mask.shape
        left_half = binary_mask[:, :w//2]
        right_half = binary_mask[:, w//2:]

        right_half_flipped = np.fliplr(right_half)

        if left_half.shape[1] != right_half_flipped.shape[1]:
            right_half_flipped = np.pad(right_half_flipped, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        intersection = np.logical_and(left_half, right_half_flipped)
        union = np.logical_or(left_half, right_half_flipped)

        iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0

        self.asymmetry = 1 - iou

    def calculate_border_irregularity(self):
        """Calcula la irregularidad del borde comparando el perímetro real con el perímetro ideal."""
        if self.perimeter > 0 and self.area > 0:
            expected_perimeter = 2 * np.sqrt(np.pi * self.area)
            self.irregularity = self.perimeter / expected_perimeter

    def calculate_simpson_index(self):
        """Calcula el índice de Simpson."""
        proportions = [p / sum(self.porcentajes_recalculados) for p in self.porcentajes_recalculados]
        simpson_index = 1 - sum([p ** 2 for p in proportions])
        return simpson_index

    def calculate_standard_deviation(self):
        """Calcula la desviación estándar de los porcentajes."""
        return np.std(self.porcentajes_recalculados)

    def display_and_save_colors(self, output_image='colores_y_mascara.png'):
        """Guarda los colores filtrados y la máscara en un archivo PNG."""
        sorted_indices = np.argsort(self.porcentajes_recalculados)[::-1]
        sorted_colores = [self.colores_filtrados[i] for i in sorted_indices]
        sorted_porcentajes = [self.porcentajes_recalculados[i] for i in sorted_indices]

        fig, axes = plt.subplots(1, len(sorted_colores) + 1, figsize=(12, 6))

        axes[0].imshow(self.binary_mask, cmap='gray')
        axes[0].set_title("Máscara Binaria")
        axes[0].axis('off')

        for i, color in enumerate(sorted_colores):
            axes[i + 1].imshow([[color]])
            axes[i + 1].axis('off')
            axes[i + 1].set_title(f"{sorted_porcentajes[i]:.2f}%")

        plt.tight_layout()
        plt.savefig(output_image)
        plt.close()
        print(f"Colores y máscara guardados en {output_image}")

    def save_colors_to_csv(self, csv_file='colores_dominantes_con_porcentaje.csv'):
        """Guarda los colores y sus porcentajes en un archivo CSV."""
        simpson_index = self.calculate_simpson_index()
        std_dev = self.calculate_standard_deviation()

        sorted_indices = np.argsort(self.porcentajes_recalculados)[::-1]
        sorted_colores = [self.colores_filtrados[i] for i in sorted_indices]
        sorted_porcentajes = [self.porcentajes_recalculados[i] for i in sorted_indices]

        file_exists = os.path.isfile(csv_file)

        self.calculate_asymmetry()
        self.calculate_border_irregularity()

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                headers = ['Imagen', 'Color 1 (RGB)', 'Porcentaje 1', 'Color 2 (RGB)', 'Porcentaje 2', 'Color 3 (RGB)', 'Porcentaje 3', 
                           'Color 4 (RGB)', 'Porcentaje 4', 'Área', 'Perímetro', 'Diámetro',
                           'Asimetría', 'Irregularidad', 'Índice Simpson', 'Desviación Estándar']
                writer.writerow(headers)

            datos_imagen = [self.image_path]
            for i, color in enumerate(sorted_colores):
                color_str = f"{color[0]},{color[1]},{color[2]}"
                datos_imagen.append(color_str)
                datos_imagen.append(f"{sorted_porcentajes[i]:.2f}%")

            datos_imagen.extend([
                f"{self.area:.2f}", f"{self.perimeter:.2f}", f"{self.diameter:.2f}",
                f"{self.asymmetry:.2f}", f"{self.irregularity:.2f}",
                f"{simpson_index:.2f}", f"{std_dev:.2f}"
            ])

            writer.writerow(datos_imagen)

        print(f"Colores y porcentajes guardados en {csv_file}")

if __name__ == "__main__":
    image_name = 'cropped_mask.png'
    analyzer = ColorAnalyzer(image_name)
    
    try:
        image = analyzer.load_and_process_image()
        analyzer.find_dominant_colors(image)
        analyzer.calculate_area_perimeter()  
        analyzer.display_and_save_colors()  
        analyzer.save_colors_to_csv()  
    except ValueError as e:
        print(e)
