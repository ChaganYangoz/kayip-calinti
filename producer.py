import cv2
import base64
import time
from kafka import KafkaProducer

# Kafka producer'ı başlat
producer = KafkaProducer(
    bootstrap_servers=['localhost:29092'],  # Docker Compose'daki PLAINTEXT_HOST ayarıyla eşleşecek şekilde değiştirildi
    value_serializer=lambda x: x.encode('utf-8'),  # Base64 encoding string olarak gönderilecek
    batch_size=16384,  # Batch boyutunu artırıyorum
    buffer_memory=33554432,  # Buffer belleğini artırıyorum (32MB)
    compression_type='gzip'  # Sıkıştırma ekliyorum
)

# Video dosyasını aç (Burada istediğin video dosyasını belirt)
video_path = 'video3.mp4'  # Buraya video dosyasının yolunu yaz
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video dosyası açılamadı!")
    exit()

# Video FPS bilgisini al
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = 1.0 / fps  # Kare başına gecikme süresi

# Gönderilen frame sayacı
frame_count = 0

# Video verilerini al ve Kafka'ya gönder
while True:
    ret, frame = cap.read()

    if not ret:
        print("Video bitmiş veya okunamıyor!")
        # Video bittiğinde özel bir mesaj gönder
        producer.send('my-topic', value="END_OF_VIDEO")
        producer.flush()  # Tüm bekleyen mesajların gönderildiğinden emin ol
        break
    
    # Frame sayacını artır
    frame_count += 1
    
    # Görüntüyü yeniden boyutlandır - performans için
    height, width = frame.shape[:2]
    new_width = 640  # Daha düşük çözünürlükte gönder
    new_height = int(height * new_width / width)
    frame = cv2.resize(frame, (new_width, new_height))

    # Görüntüyü base64 formatında encode et
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Kaliteyi 80% yap
    frame_base64 = base64.b64encode(buffer).decode('utf-8')  # Base64 string olarak dönüştür

    # Kafka'ya gönder
    producer.send('my-topic', value=frame_base64)
    
    # Her 100 frame'de bir ilerleme bilgisi göster
    if frame_count % 100 == 0:
        print(f"İşlenen frame sayısı: {frame_count}")
    
    # Videoyu doğal hızında iletmek için bekle
    time.sleep(frame_delay)

print(f"Toplam {frame_count} frame işlendi ve gönderildi.")

# Producer'ı kapat ve kaynakları serbest bırak
producer.flush()
producer.close()
cap.release()
