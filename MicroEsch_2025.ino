#include <driver/i2s.h>

// Parámetros y pines (configuración igual a la versión final de audio)
#define I2S_WS               25
#define I2S_SD               33
#define I2S_SCK              32
#define I2S_SAMPLE_BIT_COUNT 16
#define SOUND_SAMPLE_RATE    8000
#define SOUND_CHANNEL_COUNT  1
#define I2S_PORT             I2S_NUM_0

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

#define BLUE_LED             2

void setup() {
  Serial.begin(9600);
  pinMode(BLUE_LED, OUTPUT);
  digitalWrite(BLUE_LED, LOW);

  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SOUND_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
    .intr_alloc_flags = 0,
    .dma_buf_count = I2S_DMA_BUF_COUNT,
    .dma_buf_len = I2S_DMA_BUF_LEN,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  if (i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL) != ESP_OK) {
    Serial.println("Error instalando I2S");
  }
  if (i2s_set_pin(I2S_PORT, &pin_config) != ESP_OK) {
    Serial.println("Error configurando pines I2S");
  }
  if (i2s_zero_dma_buffer(I2S_PORT) != ESP_OK) {
    Serial.println("Error limpiando DMA");
  }
  if (i2s_start(I2S_PORT) != ESP_OK) {
    Serial.println("Error iniciando I2S");
  }
}

void loop() {
  size_t bytesRead = 0;
  esp_err_t res = i2s_read(I2S_PORT, (void*)StreamBuffer, StreamBufferNumBytes, &bytesRead, portMAX_DELAY);
  int samplesRead = bytesRead / sizeof(int16_t);

  if (res == ESP_OK && samplesRead > 0) {
    digitalWrite(BLUE_LED, HIGH);
    for (int i = 0; i < samplesRead; i++) {
      long amplified = (long)StreamBuffer[i] * DefAmplifyFactor;
      if (amplified > 32700) 
        amplified = 32700;
      else if (amplified < -32700) 
        amplified = -32700;
      ProcessedBuffer[i] = (int16_t)amplified;
    }
    Serial.write((uint8_t*)ProcessedBuffer, samplesRead * sizeof(int16_t));
  } else {
    digitalWrite(BLUE_LED, LOW);
  }
}
