#clase FuncUniExpress
import os  # Módulo para manejar operaciones del sistema operativo
import shutil  # Módulo para copiar y mover archivos
import numpy as np  # Biblioteca para cálculos numéricos y manejo de arreglos
import matplotlib.pyplot as plt  # Módulo para graficar imágenes y datos
from pathlib import Path  # Módulo para manejar rutas de archivos y directorios
import tensorflow as tf  # Framework de aprendizaje profundo
from tensorflow.keras.models import Sequential  # Tipo de modelo secuencial en Keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Capas de la red neuronal
from tensorflow.keras.optimizers import Adam  # Optimizador para la red neuronal
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array  # Utilidades para imágenes
from tensorflow.keras.callbacks import EarlyStopping  # Callback para detener entrenamiento temprano si no mejora
import pandas as pd  # Biblioteca para manejo de DataFrames
from PIL import Image  # Biblioteca Pillow para procesamiento de imágenes

# Desactivar mensajes innecesarios de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

class ClapModelTrainer:
    def __init__(self, 
                 dataset_path=None, 
                 model_filename='clap_model.keras', 
                 image_folder=None, 
                 labels_file='labels.txt',
                 target_size=(256, 256),  # Tamaño objetivo (alto, ancho) configurable
                 rescaled_folder=None):
        # Se define la ruta base. Si no se especifica, se utiliza la carpeta 'imagenes'
        default_path = Path(r"C:\Users\MANCH\Desktop\UTP\Semestre2025\ArqClienteServ\SongEspect\imagenes")
        self.dataset_path = Path(dataset_path) if dataset_path else default_path
        self.image_folder = Path(image_folder) if image_folder else self.dataset_path
        # Se guardan el modelo y el archivo de etiquetas en la misma carpeta base
        self.model_filename = self.dataset_path / model_filename
        self.labels_file = self.dataset_path / labels_file
        self.target_size = target_size  # Nuevo tamaño para redimensionar las imágenes
        # Carpeta donde se almacenarán las imágenes ya rescaladas
        self.rescaled_folder = Path(rescaled_folder) if rescaled_folder else self.dataset_path / "rescaled"
        self.model = None  # Inicializa el modelo en None

    def _load_labels(self):
        """Carga etiquetas de clasificación desde un archivo.
        Se espera que cada línea contenga 'nombre_imagen,etiqueta'."""
        labels = {}
        if self.labels_file.exists():
            with open(self.labels_file, 'r') as file:
                for line in file:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        labels[parts[0]] = parts[1]
        return labels

    def _save_labels(self, labels):
        """Guarda las etiquetas de clasificación en un archivo."""
        with open(self.labels_file, 'w') as file:
            for img_name, label in labels.items():
                file.write(f"{img_name},{label}\n")

    def _label_images_manually(self):
        """Permite etiquetar imágenes manualmente (si fuese necesario).
        Se utiliza para actualizar el archivo de etiquetas.
        Las imágenes se encuentran en la carpeta 'imagenes'."""
        existing_labels = self._load_labels()
        image_files = list(self.image_folder.rglob("*.png"))
        if not image_files:
            print("No se encontraron imágenes en la carpeta de imágenes.")
            return

        new_labels = {}
        for img_path in image_files:
            if img_path.name in existing_labels:
                continue

            img = load_img(img_path, color_mode='grayscale')
            img_array = img_to_array(img) / 255.0

            plt.imshow(img_array.squeeze(), cmap='gray')
            plt.title(f"Imagen: {img_path.name}")
            plt.axis("off")
            plt.show()

            while True:
                label = input("¿Esta imagen es un APLAUSO o OTRO SONIDO? (aplauso/otro): ").strip().lower()
                if label in {"aplauso", "otro"}:
                    break
                print("Entrada no válida. Por favor, ingresa 'aplauso' o 'otro'.")

            new_labels[img_path.name] = label

        existing_labels.update(new_labels)
        self._save_labels(existing_labels)
        print("Etiquetado manual completado y guardado.")

    def _prepare_rescaled_images(self):
        """
        Preprocesa y guarda imágenes redimensionadas en la carpeta 'rescaled'
        para acelerar el entrenamiento, utilizando interpolación LANCZOS para
        minimizar la pérdida de información.
        """
        if not self.rescaled_folder.exists():
            self.rescaled_folder.mkdir(parents=True, exist_ok=True)

        # Si ya existen imágenes redimensionadas, se utilizan directamente
        rescaled_files = list(self.rescaled_folder.glob("*.png"))
        if rescaled_files:
            print("Imágenes rescaladas ya existentes. Se utilizarán esas imágenes.")
            return

        image_files = list(self.image_folder.glob("*.png"))
        if not image_files:
            print("No se encontraron imágenes en la carpeta original.")
            return

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    # Convertir a escala de grises (modo 'L') para asegurar consistencia
                    img = img.convert('L')
                    # Redimensionar usando LANCZOS, que preserva la calidad
                    # Nota: Image.resize recibe tamaño en (ancho, alto)
                    img_rescaled = img.resize((self.target_size[1], self.target_size[0]), resample=Image.LANCZOS)
                    # Guardar la imagen redimensionada en la carpeta 'rescaled'
                    save_path = self.rescaled_folder / img_path.name
                    img_rescaled.save(save_path)
            except Exception as e:
                print(f"Error al procesar {img_path.name}: {e}")
        print("Proceso de redimensionamiento completado.")

    def _create_model(self, input_shape):
        """Crea y compila un modelo de red neuronal basado en una CNN para extraer características de las imágenes."""
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.25),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_and_evaluate(self):
        """
        Entrena y evalúa el modelo usando las imágenes redimensionadas de la carpeta 'rescaled'.
        Se utiliza un DataFrame que asocia cada imagen con su etiqueta (cargada desde 'labels.txt'),
        y se divide en entrenamiento y validación.
        """
        labels = self._load_labels()
        # Prepara las imágenes redimensionadas (si aún no se han generado)
        self._prepare_rescaled_images()

        # Se obtienen las imágenes de la carpeta 'rescaled' que tengan etiqueta
        file_list = []
        classes = []
        for img_path in self.rescaled_folder.glob("*.png"):
            if img_path.name in labels:
                file_list.append(img_path.name)
                classes.append(labels[img_path.name])
        if not file_list:
            print("No se encontraron imágenes etiquetadas en la carpeta rescalada.")
            return

        # Crear un DataFrame con los nombres de archivos y sus clases
        df = pd.DataFrame({'filename': file_list, 'class': classes})
        datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0 / 255)

        train_gen = datagen.flow_from_dataframe(
            dataframe=df,
            directory=str(self.rescaled_folder),
            x_col='filename',
            y_col='class',
            target_size=self.target_size,  # Las imágenes ya están redimensionadas al tamaño deseado
            color_mode='grayscale',
            class_mode='binary',
            batch_size=8,
            subset='training',
            shuffle=True
        )

        val_gen = datagen.flow_from_dataframe(
            dataframe=df,
            directory=str(self.rescaled_folder),
            x_col='filename',
            y_col='class',
            target_size=self.target_size,
            color_mode='grayscale',
            class_mode='binary',
            batch_size=8,
            subset='validation',
            shuffle=False
        )

        # El input_shape se define en función del tamaño objetivo (alto, ancho, canales)
        input_shape = (self.target_size[0], self.target_size[1], 1)
        model = self._create_model(input_shape)

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=[early_stop]
        )

        model.save(self.model_filename)
        print(f"Modelo guardado en {self.model_filename}")

        loss, accuracy = model.evaluate(val_gen)
        print(f"Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%")
        self.model = model

    def test_model(self, test_folder_path):
        """
        Prueba el modelo entrenado con imágenes de una carpeta dada.
        Para cada imagen se realiza el preprocesamiento (conversión a escala de grises,
        redimensionamiento y normalización) y se obtiene la predicción, mostrando el resultado.
        """
        # Cargar el modelo si aún no está cargado
        if self.model is None:
            if self.model_filename.exists():
                self.model = tf.keras.models.load_model(self.model_filename)
                print("Modelo cargado desde archivo.")
            else:
                print("No se encontró el modelo guardado.")
                return

        test_folder = Path(test_folder_path)
        if not test_folder.exists():
            print("La carpeta de prueba no existe.")
            return

        image_files = list(test_folder.glob("*.png"))
        if not image_files:
            print("No se encontraron imágenes en la carpeta de prueba.")
            return

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    # Convertir a escala de grises y redimensionar usando LANCZOS
                    img = img.convert('L')
                    img_resized = img.resize((self.target_size[1], self.target_size[0]), resample=Image.LANCZOS)
                    # Convertir la imagen a array y normalizar
                    img_array = img_to_array(img_resized) / 255.0
                    # Expandir dimensiones para que sea compatible con el modelo (batch size 1)
                    img_array = np.expand_dims(img_array, axis=0)
                    prediction = self.model.predict(img_array)
                    # Con 'binary' mode: "aplauso" es 0 y "otro" es 1
                    label_pred = "otro" if prediction[0][0] > 0.5 else "aplauso"
                    
                    plt.imshow(np.squeeze(img_array), cmap='gray')
                    plt.title(f"Predicción: {label_pred}")
                    plt.axis("off")
                    plt.show()
            except Exception as e:
                print(f"Error al procesar {img_path.name}: {e}")

if __name__ == "__main__":
    # Instanciar y entrenar el modelo
    trainer = ClapModelTrainer(target_size=(256,256))
    # Si es necesario, se puede etiquetar manualmente para actualizar 'labels.txt'
    # trainer._label_images_manually()
    trainer.train_and_evaluate()
    
    # Ruta de la carpeta de prueba (ajusta la ruta si es necesario)
    test_folder = r"C:\Users\MANCH\Desktop\UTP\Semestre2025\ArqClienteServ\SongEspect\imagenes"
    trainer.test_model(test_folder)
