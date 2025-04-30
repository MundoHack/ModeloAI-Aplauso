import os
import sys
import time
import wave
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pyaudio
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import serial  # Comunicación serial con la ESP32
import tensorflow as tf
from pathlib import Path

# Se asume que la clase ClapModelTrainer contiene funciones para preparar imágenes
# pero ya no se entrenará el modelo, solo se utilizará para cargarlo y predecir.
class ClapModelTrainer:
    def __init__(self, 
                 dataset_path=None, 
                 model_filename='clap_model.keras', 
                 image_folder=None, 
                 labels_file='labels.txt',
                 target_size=(256, 256),
                 rescaled_folder=None):
        default_path = Path(r"C:\Users\MANCH\Desktop\UTP\Semestre2025\ArqClienteServ\SongEspect\imagenes")
        self.dataset_path = Path(dataset_path) if dataset_path else default_path
        self.image_folder = Path(image_folder) if image_folder else self.dataset_path
        self.model_filename = self.dataset_path / model_filename
        self.labels_file = self.dataset_path / labels_file
        self.target_size = target_size
        self.rescaled_folder = Path(rescaled_folder) if rescaled_folder else self.dataset_path / "rescaled"
        self.model = None

    def _load_labels(self):
        labels = {}
        if self.labels_file.exists():
            with open(self.labels_file, 'r') as file:
                for line in file:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        labels[parts[0]] = parts[1]
        return labels

    def _prepare_rescaled_images(self):
        if not self.rescaled_folder.exists():
            self.rescaled_folder.mkdir(parents=True, exist_ok=True)
        rescaled_files = list(self.rescaled_folder.glob("*.png"))
        if rescaled_files:
            print("Imágenes rescaladas ya existentes. Se utilizarán esas imágenes.")
            return
        image_files = list(self.image_folder.glob("*.png"))
        if not image_files:
            print("No se encontraron imágenes en la carpeta original.")
            return
        from PIL import Image
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    img = img.convert('L')
                    img_rescaled = img.resize((self.target_size[1], self.target_size[0]), resample=Image.LANCZOS)
                    save_path = self.rescaled_folder / img_path.name
                    img_rescaled.save(save_path)
            except Exception as e:
                print(f"Error al procesar {img_path.name}: {e}")
        print("Proceso de redimensionamiento completado.")

    # Se elimina el método de entrenamiento ya que el modelo se carga desde archivo.
    # def train_and_evaluate(self):
    #     ...

# Parámetros de grabación
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 3000         # Umbral para detectar sonido
PRE_RECORD_DURATION = 1  # Segundos previos a guardar
DURATION = 2             # Segundos a grabar tras detectar el sonido

# Carpeta de destino para guardar audio y espectrograma
SAVE_DIR = r"C:\Users\MANCH\Desktop\UTP\Semestre2025\ArqClienteServ\SongEspect"
os.makedirs(SAVE_DIR, exist_ok=True)

# Instanciar el objeto y cargar el modelo entrenado
trainer = ClapModelTrainer(target_size=(256, 256))
if trainer.model is None:
    if trainer.model_filename.exists():
        trainer.model = tf.keras.models.load_model(trainer.model_filename)
        print("Modelo cargado para predicción.")
    else:
        print("No se encontró un modelo entrenado. Primero entrena el modelo y guárdalo en:")
        print(trainer.model_filename)
        sys.exit(1)

num_events = int(input("Ingrese el número de sonidos a detectar: "))

# Configurar comunicación serial con la ESP32 (ajusta el puerto y baud rate según corresponda)
try:
    esp_serial = serial.Serial("COM8", 115200, timeout=1)
    print("Conexión con ESP32 establecida.")
except Exception as e:
    print(f"Error al conectar con ESP32: {e}")
    esp_serial = None

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("🔊 Escuchando...")
event_count = 0
pre_buffer = deque(maxlen=int(RATE / CHUNK * PRE_RECORD_DURATION))

def save_audio(frames, filename):
    filepath = os.path.join(SAVE_DIR, filename)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print(f"🎵 Audio guardado como '{filepath}'")
    return filepath

def save_spectrogram(audio_data, filename):
    filepath = os.path.join(SAVE_DIR, filename)
    y = np.array(audio_data, dtype=np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    S = librosa.feature.melspectrogram(y=y, sr=RATE, n_mels=256, fmax=16000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=RATE, x_axis='time', y_axis='mel', cmap='gray')
    plt.axis('off')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"📸 Espectrograma guardado como '{filepath}'")
    return filepath

start_time = time.time()
while event_count < num_events:
    elapsed_time = int(time.time() - start_time)
    sys.stdout.write(f"\r⏳ Tiempo esperando: {elapsed_time} segundos")
    sys.stdout.flush()
    
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    pre_buffer.append(data)
    peak = np.max(np.abs(data))
    
    if peak > THRESHOLD:
        event_count += 1
        print(f"\n👏 Sonido {event_count} detectado! Grabando...")
        frames = list(pre_buffer)  # Incluir audio previo
        for _ in range(0, int(RATE / CHUNK * DURATION)):
            frames.append(np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16))
        
        audio_filename = f"Sonido_{event_count}.wav"
        spectrogram_filename = f"Sonido_{event_count}.png"
        
        audio_filepath = save_audio(frames, audio_filename)
        audio_data = np.concatenate(frames)
        spectrogram_filepath = save_spectrogram(audio_data, spectrogram_filename)
        
        # Preprocesar la imagen del espectrograma para la predicción
        img = load_img(spectrogram_filepath, color_mode='grayscale', target_size=trainer.target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = trainer.model.predict(img_array)
        label_pred = "otro" if prediction[0][0] > 0.5 else "aplauso"
        print(f"🔍 Predicción: {label_pred}")
        
        # Si se predice "aplauso", enviar señal a la ESP32 para encender el LED
        if label_pred == "aplauso" and esp_serial is not None:
            try:
                esp_serial.write(b'LED_ON\n')
                print("✅ Señal enviada a la ESP32 para encender el LED D4.")
            except Exception as e:
                # En caso de error, se envía el comando de apagado
                print(f"Error al enviar señal a la ESP32: {e}")
        
        start_time = time.time()  # Reiniciar tiempo de espera para el siguiente evento

stream.stop_stream()
stream.close()
p.terminate()

if esp_serial is not None:
    esp_serial.close()
    print("Conexión con ESP32 cerrada.")

print("✅ Proceso finalizado.")
