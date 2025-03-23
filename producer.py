import cv2
import base64
from kafka import KafkaProducer

# Kafka producer'ı başlat
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: x.encode('utf-8')  # Base64 encoding string olarak gönderilecek
)

# Video dosyasını aç (Burada istediğin video dosyasını belirt)
video_path = 'video2.mp4'  # Buraya video dosyasının yolunu yaz
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video dosyası açılamadı!")
    exit()

# Video verilerini al ve Kafka'ya gönder
while True:
    ret, frame = cap.read()

    if not ret:
        print("Video bitmiş veya okunamıyor!")
        break

    # Görüntüyü base64 formatında encode et
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')  # Base64 string olarak dönüştür

    # Kafka'ya gönder
    producer.send('my-topic', value=frame_base64)

    # Video'yu ekranda göster (isteğe bağlı)
    cv2.imshow('Video Oynatılıyor', frame)

    # 'q' tuşuna basarak çıkabilirsin
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
