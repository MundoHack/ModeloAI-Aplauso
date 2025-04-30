import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav
import os

class AudioProcessor:
    def __init__(self, duration=2, samplerate=22050):
        self.duration = duration
        self.samplerate = samplerate
        self.audio_folder = "audios"
        self.image_folder = "imagenes"

        # Crear carpetas si no existen
        os.makedirs(self.audio_folder, exist_ok=True)
        os.makedirs(self.image_folder, exist_ok=True)

    def record_audio(self, filename):
        """ Graba audio y lo guarda en un archivo WAV en la carpeta 'audios/' """
        filepath = os.path.join(self.audio_folder, filename)

        print(f"Grabando {filename}...")
        audio = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=1, dtype='float32')
        sd.wait()
        wav.write(filepath, self.samplerate, (audio * 32767).astype(np.int16))
        print(f"Grabación guardada en {filepath}")

    def detect_applause(self, audio_file):
        """ Detecta aplausos basándose en picos de energía en el audio """
        audio_path = os.path.join(self.audio_folder, audio_file)
        y, sr = librosa.load(audio_path, sr=self.samplerate)

        # Obtener el envelope de energía
        energy = librosa.feature.rms(y=y)[0]
        times = librosa.times_like(energy, sr=sr)

        # Detectar picos de energía
        threshold = np.percentile(energy, 95)  # Umbral dinámico
        peaks = times[energy > threshold]

        return peaks

    def generate_spectrogram(self, audio_file, output_image):
        """ Genera un espectrograma con ejes de tiempo y frecuencia, y marca los aplausos detectados """
        audio_path = os.path.join(self.audio_folder, audio_file)
        image_path = os.path.join(self.image_folder, output_image)

        y, sr = librosa.load(audio_path, sr=self.samplerate)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Crear figura
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='gray')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Espectrograma en escala de grises")

        # Detectar aplausos y marcarlos
        peaks = self.detect_applause(audio_file)
        plt.vlines(peaks, ymin=0, ymax=sr//2, color='red', linestyle='--', label="Aplauso detectado")
        plt.legend()

        # Guardar la imagen
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Espectrograma guardado en {image_path}")

if __name__ == "__main__":
    processor = AudioProcessor()
    num_aplausos = int(input("¿Cuántos aplausos deseas grabar? "))
    
    for i in range(1, num_aplausos + 1):
        filename = f"aplauso_{i}.wav"
        processor.record_audio(filename)
        image_filename = f"espectrograma_{i}.png"
        processor.generate_spectrogram(filename, image_filename)
