import socket

# ⚡ Configurar el servidor
HOST = "0.0.0.0"  # Acepta conexiones desde cualquier IP en la red local
PORT = 12345      # Puerto de comunicación (debe coincidir con el de la ESP32)

def iniciar_servidor():
    """Inicia el servidor TCP que espera conexiones de la ESP32 y recibe datos de audio."""
    try:
        # 1️⃣ Crear el socket del servidor
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((HOST, PORT))  # Enlazar el servidor a la IP y puerto
        server_socket.listen(1)           # Permitir solo una conexión a la vez

        print(f"✅ Servidor escuchando en {HOST}:{PORT}")

        # 2️⃣ Esperar conexión de la ESP32
        client_socket, client_address = server_socket.accept()
        print(f"🔗 Conexión establecida con {client_address}")

        # 3️⃣ Leer el mensaje de handshake de la ESP32
        data = client_socket.recv(1024).decode().strip()
        print(f"📩 Mensaje recibido: {data}")

        if data == "conectado":
            response = "encender led\n"
            client_socket.send(response.encode())  # 4️⃣ Responder a la ESP32
            print(f"📤 Enviando: {response.strip()}")
        
        # 5️⃣ Recibir datos de audio de forma continua
        print("🎙️ Recibiendo datos de audio...")
        with open("audio.raw", "wb") as audio_file:
            while True:
                # Se lee un bloque de datos (ajusta el tamaño de búfer si es necesario)
                audio_data = client_socket.recv(4096)
                if not audio_data:
                    # Si no se reciben datos, se cierra la conexión
                    print("❌ No se han recibido más datos. Cerrando conexión.")
                    break
                # Escribir los datos binarios recibidos en el archivo
                audio_file.write(audio_data)
                print(f"📥 Recibidos {len(audio_data)} bytes")
        
        # 6️⃣ Cerrar la conexión
        client_socket.close()
        server_socket.close()
        print("❌ Conexión cerrada.")
    
    except Exception as e:
        print(f"❌ Error en el servidor: {e}")

if __name__ == "__main__":
    iniciar_servidor()
