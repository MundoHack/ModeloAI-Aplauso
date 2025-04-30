import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ClapPredictor:
    def __init__(self, model_filename='clap_model.keras', image_folder=None):
        """Carga el modelo y define la ruta de imágenes."""
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"El modelo '{model_filename}' no se encontró.")
        
        self.model = tf.keras.models.load_model(model_filename)
        print(f"Modelo '{model_filename}' cargado correctamente.")

        # Ruta de la carpeta de imágenes
        if image_folder is None:
            image_folder = r'C:\Users\MANCH\Desktop\UTP\Semestre2025\ArqClienteServ\SongEspect\imagenes'
        
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"La carpeta de imágenes '{image_folder}' no existe.")
        
        self.image_folder = image_folder

    def predict_all(self):
        """Predice todas las imágenes en la carpeta que comiencen con 'espectrograma_'."""
        image_files = [f for f in os.listdir(self.image_folder) if f.startswith("espectrograma_") and f.endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            raise FileNotFoundError(f"No se encontraron imágenes 'espectrograma_*' en {self.image_folder}")

        for image_name in image_files:
            image_path = os.path.join(self.image_folder, image_name)
            self.predict(image_path)

    def predict(self, spectrogram_image):
        """ Predice si un espectrograma representa un aplauso u otro sonido """
        if not os.path.exists(spectrogram_image):
            raise FileNotFoundError(f"La imagen '{spectrogram_image}' no se encontró.")

        img = load_img(spectrogram_image, target_size=(128, 74), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)[0][0]
        confidence = prediction * 100

        if prediction > 0.5:
            print(f"{os.path.basename(spectrogram_image)}: Es un aplauso ({confidence:.2f}% de confianza)")
        else:
            print(f"{os.path.basename(spectrogram_image)}: Otro sonido ({100 - confidence:.2f}% de confianza)")

if __name__ == "__main__":
    predictor = ClapPredictor()
    predictor.predict_all()
