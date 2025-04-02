#define BLUETOOTH "ESP32BT"
#include "esp32dumbdisplay.h"
#include <driver/i2s.h>

// Inicialización de la interfaz DumbDisplay vía Bluetooth
DumbDisplay dumbdisplay(new DDBluetoothSerialIO(BLUETOOTH));

// Asignación de pines I2S para INMP441
#define I2S_WS 25
#define I2S_SD 33
#define I2S_SCK 32
#define I2S_SAMPLE_BIT_COUNT 16
#define SOUND_SAMPLE_RATE 8000
#define SOUND_CHANNEL_COUNT 1
#define I2S_PORT I2S_NUM_0

// Declaración de capas para la interfaz
PlotterDDLayer* plotterLayer;
LcdDDLayer* micTabLayer;
LcdDDLayer* recTabLayer;
LcdDDLayer* playTabLayer;
LcdDDLayer* startBtnLayer;
LcdDDLayer* stopBtnLayer;
LcdDDLayer* amplifyLblLayer;
LedGridDDLayer* amplifyMeterLayer;

// Nombre del archivo WAV grabado (siempre se sobrescribe)
const char* SoundName = "recorded_sound";

// Parámetros DMA para I2S
const int I2S_DMA_BUF_COUNT = 8;
const int I2S_DMA_BUF_LEN = 1024;

#if I2S_SAMPLE_BIT_COUNT == 32
const int StreamBufferNumBytes = 512;
const int StreamBufferLen = StreamBufferNumBytes / 4;
int32_t StreamBuffer[StreamBufferLen];
#else
#if SOUND_SAMPLE_RATE == 16000
// Para 16 bits y 16000 muestras por segundo: 512 bytes (16 ms de lectura)
const int StreamBufferNumBytes = 512;
#else
// Para 16 bits y 8000 muestras por segundo: 256 bytes (16 ms de lectura)
const int StreamBufferNumBytes = 256;
#endif
const int StreamBufferLen = StreamBufferNumBytes / 2;
int16_t StreamBuffer[StreamBufferLen];
int16_t ProcessedBuffer[StreamBufferLen];
#endif

// Amplificación de muestras de sonido (16 bits)
const int MaxAmplifyFactor = 20;
const int DefAmplifyFactor = 10;

// Prototipos de funciones para I2S
esp_err_t i2s_install();
esp_err_t i2s_setpin();

// Variables de estado globales
DDConnectVersionTracker cvTracker;  // Rastrea la conexión establecida con DD
int what = 1;                       // 1: micrófono; 2: grabar; 3: reproducir
bool started = false;
int amplifyFactor = DefAmplifyFactor;
int soundChunkId = -1;              // Id asignado al comenzar a enviar sonido
long streamingMillis = 0;
int streamingTotalSampleCount = 0;

void setup() {

  Serial.begin(115200);
  Serial.println("CONFIGURANDO MICRÓFONO ...");

  // Configurar I2S
  if (i2s_install() != ESP_OK) {
    Serial.println("XXX falló al instalar I2S");
  }
  if (i2s_setpin() != ESP_OK) {
    Serial.println("XXX falló al configurar los pines I2S");
  }
  if (i2s_zero_dma_buffer(I2S_PORT) != ESP_OK) {
    Serial.println("XXX falló al poner a cero el buffer DMA de I2S");
  }
  if (i2s_start(I2S_PORT) != ESP_OK) {
    Serial.println("XXX falló al iniciar I2S");
  }

  Serial.println("... MICRÓFONO CONFIGURADO");

  // Grabar comandos de configuración de la interfaz
  dumbdisplay.recordLayerSetupCommands();

  // Crear capa para graficar la señal de audio
  plotterLayer = dumbdisplay.createPlotterLayer(1024, 256, SOUND_SAMPLE_RATE / StreamBufferLen);

  // Crear capas LCD para las pestañas "MIC", "REC" y "PLAY"
  micTabLayer = dumbdisplay.createLcdLayer(8, 1);
  micTabLayer->writeCenteredLine("MIC");
  micTabLayer->border(1, "gray");
  micTabLayer->enableFeedback("f");

  recTabLayer = dumbdisplay.createLcdLayer(8, 1);
  recTabLayer->writeCenteredLine("REC");
  recTabLayer->border(1, "gray");
  recTabLayer->enableFeedback("f");

  playTabLayer = dumbdisplay.createLcdLayer(8, 1);
  playTabLayer->writeCenteredLine("PLAY");
  playTabLayer->border(1, "gray");
  playTabLayer->enableFeedback("f");

  // Crear capas LCD para los botones "START" y "STOP"
  startBtnLayer = dumbdisplay.createLcdLayer(12, 3);
  startBtnLayer->pixelColor("darkgreen");
  startBtnLayer->border(2, "darkgreen", "round");
  startBtnLayer->margin(1);
  startBtnLayer->enableFeedback("fl");

  stopBtnLayer = dumbdisplay.createLcdLayer(12, 3);
  stopBtnLayer->pixelColor("darkred");
  stopBtnLayer->border(2, "darkgreen", "round");
  stopBtnLayer->margin(1);
  stopBtnLayer->enableFeedback("fl");

  // Crear etiqueta para el medidor de amplificación
  amplifyLblLayer = dumbdisplay.createLcdLayer(12, 1);
  amplifyLblLayer->pixelColor("darkred");
  amplifyLblLayer->noBackgroundColor();

  // Crear capa para el medidor de amplificación (barra LED)
  amplifyMeterLayer = dumbdisplay.createLedGridLayer(MaxAmplifyFactor, 1, 1, 2);
  amplifyMeterLayer->onColor("darkblue");
  amplifyMeterLayer->offColor("lightgray");
  amplifyMeterLayer->border(0.2, "blue");
  amplifyMeterLayer->enableFeedback("fa:rpt50");

  // Organizar la disposición de las capas usando DDAutoPinConfig
  DDAutoPinConfig builder('V');  // 'V' para disposición vertical
  builder
    .addLayer(plotterLayer)
    .beginGroup('H')  // grupo horizontal
      .addLayer(micTabLayer)
      .addLayer(recTabLayer)
      .addLayer(playTabLayer)
    .endGroup()
    .beginGroup('H')  // grupo horizontal para botones
      .addLayer(startBtnLayer)
      .addLayer(stopBtnLayer)
    .endGroup()
    .beginGroup('S')  // grupo apilado (stack)
      .addLayer(amplifyLblLayer)
      .addLayer(amplifyMeterLayer)
    .endGroup();

  dumbdisplay.configAutoPin(builder.build());

  // Reproducir y persistir la configuración de la interfaz para reconexiones
  dumbdisplay.playbackLayerSetupCommands("esp32ddmice");

  // Establecer callback de inactividad
  dumbdisplay.setIdleCallback([](long idleForMillis, DDIdleConnectionState connectionState) {
    if (connectionState == DDIdleConnectionState::IDLE_RECONNECTING) {
      started = false;
    }
  });
}

void loop() {

  bool updateTab = false;
  bool updateStartStop = false;
  bool updateAmplifyFactor = false;

  if (cvTracker.checkChanged(dumbdisplay)) {
    // Si la conexión ha cambiado, reiniciar estado y actualizar UI
    started = false;
    updateTab = true;
    updateStartStop = true;
    updateAmplifyFactor = true;
  } else {
    // Verificar cambios en la selección de pestaña
    int oriWhat = what;
    if (micTabLayer->getFeedback()) {
      what = 1;
    } else if (recTabLayer->getFeedback()) {
      what = 2;
    } else if (playTabLayer->getFeedback()) {
      what = 3;
    }
    if (what != oriWhat) {
      started = false;
      updateTab = true;
      updateStartStop = true;
    }
    if (startBtnLayer->getFeedback()) {
      started = true;
      updateStartStop = true;
    } else if (stopBtnLayer->getFeedback()) {
      started = false;
      updateStartStop = true;
    }
    const DDFeedback* feedback = amplifyMeterLayer->getFeedback();
    if (feedback != NULL) {
      amplifyFactor = feedback->x + 1;
      updateAmplifyFactor = true;
    }
  }

  // Actualizar pestañas según el modo seleccionado
  if (updateTab) {
    const char* micColor = what == 1 ? "blue" : "gray";
    const char* micBoarderShape = what == 1 ? "flat" : "hair";
    const char* recColor = what == 2 ? "blue" : "gray";
    const char* recBoarderShape = what == 2 ? "flat" : "hair";
    const char* playColor = what == 3 ? "blue" : "gray";
    const char* playBoarderShape = what == 3 ? "flat" : "hair";
    micTabLayer->border(1, micColor, micBoarderShape);
    micTabLayer->pixelColor(micColor);
    recTabLayer->border(1, recColor, recBoarderShape);
    recTabLayer->pixelColor(recColor);
    playTabLayer->border(1, playColor, playBoarderShape);
    playTabLayer->pixelColor(playColor);
  }

  // Actualizar botones de inicio/parada según el estado
  if (updateStartStop) {
    const char* whatTitle;
    if (what == 1) {
      whatTitle = "MIC";
    } else if (what == 2) {
      whatTitle = "REC";
    } else { // what == 3
      whatTitle = "PLAY";
    }
    startBtnLayer->writeCenteredLine(String("Start ") + whatTitle, 1);
    stopBtnLayer->writeCenteredLine(String("Stop ") + whatTitle, 1);
    if (what == 3) {
      startBtnLayer->disabled(false);
      stopBtnLayer->disabled(false);
      amplifyMeterLayer->disabled(true);
    } else {
      if (started) {
        startBtnLayer->disabled(true);
        stopBtnLayer->disabled(false);
      } else {
        startBtnLayer->disabled(false);
        stopBtnLayer->disabled(true);
      }
      micTabLayer->disabled(started);
      recTabLayer->disabled(started);
      playTabLayer->disabled(started);
      amplifyMeterLayer->disabled(false);
    }
  }

  // Actualizar medidor de amplificación
  if (updateAmplifyFactor) {
    amplifyMeterLayer->horizontalBar(amplifyFactor);
    amplifyLblLayer->writeLine(String(amplifyFactor), 0, "R");
  }

  // Leer datos I2S y procesar la señal
  size_t bytesRead = 0;
  esp_err_t result = i2s_read(I2S_PORT, &StreamBuffer, StreamBufferNumBytes, &bytesRead, portMAX_DELAY);
  int samplesRead = 0;

#if I2S_SAMPLE_BIT_COUNT == 32
  int16_t sampleStreamBuffer[StreamBufferLen];
#else
  int16_t* sampleStreamBuffer = StreamBuffer;
#endif

  if (result == ESP_OK) {
#if I2S_SAMPLE_BIT_COUNT == 32
    samplesRead = bytesRead / 4;
#else
    samplesRead = bytesRead / 2;
#endif
    if (samplesRead > 0) {
      float sumVal = 0;
      for (int i = 0; i < samplesRead; ++i) {
        int32_t val = StreamBuffer[i];
#if I2S_SAMPLE_BIT_COUNT == 32
        val = val / 0x0000ffff;
#endif
        if (amplifyFactor > 1) {
          val = amplifyFactor * val;
          if (val > 32700) {
            val = 32700;
          } else if (val < -32700) {
            val = -32700;
          }
        }
        sampleStreamBuffer[i] = val;
        sumVal += val;
      }
      float meanVal = sumVal / samplesRead;
      plotterLayer->set(meanVal);
    }
  }

  // Modo PLAY: reproducir o detener sonido según feedback
  if (what == 3) {
    if (updateStartStop) {
      if (started) {
        dumbdisplay.playSound(SoundName);
      } else {
        dumbdisplay.stopSound();
      }
    }
    return;
  }

  // En modos MIC y REC, gestionar el envío/almacenamiento de la señal
  if (started) {
    if (soundChunkId == -1) {
      if (what == 1) {
        soundChunkId = dumbdisplay.streamSound16(SOUND_SAMPLE_RATE, SOUND_CHANNEL_COUNT);
        dumbdisplay.writeComment(String("INICIADO streaming de micrófono con id de chunk [") + soundChunkId + "]");
      } else if (what == 2) {
        soundChunkId = dumbdisplay.saveSoundChunked16(SoundName, SOUND_SAMPLE_RATE, SOUND_CHANNEL_COUNT);
        dumbdisplay.writeComment(String("INICIADO grabación de streaming con id de chunk [") + soundChunkId + "]");
      }
      streamingMillis = millis();
      streamingTotalSampleCount = 0;
    }
  }

  if (result == ESP_OK && soundChunkId != -1) {
    bool isFinalChunk = !started;
    dumbdisplay.sendSoundChunk16(soundChunkId, sampleStreamBuffer, samplesRead, isFinalChunk);
    streamingTotalSampleCount += samplesRead;
    if (isFinalChunk) {
      dumbdisplay.writeComment(String("FINALIZADO streaming con id de chunk [") + soundChunkId + "]");
      long forMillis = millis() - streamingMillis;
      int totalSampleCount = streamingTotalSampleCount;
      dumbdisplay.writeComment(String(". total de muestras transmitidas: ") + totalSampleCount + " en " + String(forMillis / 1000.0) + "s");
      dumbdisplay.writeComment(String(". tasa de muestras de transmisión: ") + String(1000.0 * ((float)totalSampleCount / forMillis)));
      soundChunkId = -1;
    }
  }
}

esp_err_t i2s_install() {
  uint32_t mode = I2S_MODE_MASTER | I2S_MODE_RX;
#if I2S_SCK == I2S_PIN_NO_CHANGE
  mode |= I2S_MODE_PDM;
#endif
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(mode),
    .sample_rate = SOUND_SAMPLE_RATE,
    .bits_per_sample = i2s_bits_per_sample_t(I2S_SAMPLE_BIT_COUNT),
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
    .intr_alloc_flags = 0,
    .dma_buf_count = I2S_DMA_BUF_COUNT,
    .dma_buf_len = I2S_DMA_BUF_LEN,
    .use_apll = false
  };
  return i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

esp_err_t i2s_setpin() {
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };
  return i2s_set_pin(I2S_PORT, &pin_config);
}
