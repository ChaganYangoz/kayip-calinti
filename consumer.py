import cv2
import base64
import numpy as np
import torch
from kafka import KafkaConsumer
import time
from collections import Counter
import warnings

# FutureWarning uyarılarını bastır
warnings.filterwarnings("ignore", category=FutureWarning)

# Renk isimlerine karar vermek için fonksiyon
def get_color_name(rgb):
    # Yaygın renk isimlerini ve RGB değerlerini tanımla
    colors = {
        'kırmızı': (255, 0, 0),
        'yeşil': (0, 255, 0),
        'mavi': (0, 0, 255),
        'sarı': (255, 255, 0),
        'mor': (128, 0, 128),
        'turuncu': (255, 165, 0),
        'pembe': (255, 192, 203),
        'kahverengi': (165, 42, 42),
        'siyah': (0, 0, 0),
        'beyaz': (255, 255, 255),
        'gri': (128, 128, 128)
    }
    
    # RGB değerlerinizi BGR'den RGB'ye dönüştür
    b, g, r = rgb
    rgb = (r, g, b)
    
    # En yakın rengi bul
    min_distance = float('inf')
    closest_color = "bilinmeyen"
    
    for color_name, color_rgb in colors.items():
        distance = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    return closest_color

# Bir görüntüdeki baskın rengi bulma fonksiyonu
def get_dominant_color(image, k=5):
    # Resmi düzleştir
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # K-means algoritması için durma kriterleri
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # K-means uygula
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # En çok kullanılan renk merkezi
    counts = Counter(labels.flatten())
    dominant_color = centers[counts.most_common(1)[0][0]]
    
    # BGR formatında döndür
    return tuple(map(int, dominant_color))

# PyTorch modelini yükleme öncesi import etmek için
from torch import amp

# YOLOv5 modelini yükle
# Orijinal torch.hub.load kullanımı yerine doğrudan repo URL'si ile yükleme
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=False)  

# YOLOv5 repo dosyasını düzelt (isteğe bağlı - gerekirse kullanın)
"""
# Bu kod bloğunu çalıştırmak isterseniz, yorum işaretlerini kaldırın
import os
common_py_path = os.path.join(os.path.expanduser('~'), '.cache/torch/hub/ultralytics_yolov5_master/models/common.py')

if os.path.exists(common_py_path):
    with open(common_py_path, 'r') as file:
        content = file.read()
    
    # Eski kodu yeni kodla değiştir
    content = content.replace('with amp.autocast(autocast):', 'with torch.amp.autocast(\'cuda\', enabled=autocast):')
    
    with open(common_py_path, 'w') as file:
        file.write(content)
    print("YOLOv5 common.py dosyası güncellendi.")
"""

# Kafka consumer'ı başlat
consumer = KafkaConsumer(
    'my-topic',  # Kafka topic adı
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: x.decode('utf-8')  # Base64 string olarak deserialize et
)

# Son işlenen zaman (frame)
last_processed_time = time.time()

# Kafka'dan gelen veriyi işleme döngüsü
for message in consumer:
    # Kafka'dan gelen base64 string verisini al
    frame_base64 = message.value

    # Base64 verisini çöz
    frame_bytes = base64.b64decode(frame_base64)

    # Çözülmüş byte verisini NumPy array'e çevir
    np_array = np.frombuffer(frame_bytes, dtype=np.uint8)

    # NumPy array'ini OpenCV formatına dönüştür
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Geçen zaman
    elapsed_time = time.time() - last_processed_time

    # Her 1 saniyede bir frame işleme (zaman kontrolü)
    if elapsed_time >= 1:  # 1 saniyede bir frame işleme
        # YOLOv5 ile nesne tespiti yap
        results = model(frame)  # Görüntüyü YOLOv5 modeline ver
        
        # Sonuçları pandas DataFrame olarak al
        df = results.pandas().xyxy[0]
        
        # Orijinal görüntünün bir kopyasını oluştur
        output_frame = frame.copy()
        
        # Her tespit edilen nesne için
        for i, row in df.iterrows():
            # Koordinatları ve etiketi al
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            conf = row['confidence']
            
            # Nesnenin bulunduğu bölgeyi kes
            roi = frame[ymin:ymax, xmin:xmax]
            
            # Eğer ROI boş değilse
            if roi.size > 0:
                # Nesnenin baskın rengini bul
                dominant_color = get_dominant_color(roi)
                color_name = get_color_name(dominant_color)
                
                # Metin ve kutu çizimi
                # Tespit edilen nesne etrafına kutu çiz
                cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), dominant_color, 2)
                
                # Etiket metni oluştur
                text = f"{label} ({color_name}) {conf:.2f}"
                
                # Metin için kutu çiz
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_bg_coord1 = (xmin, ymin - 20)
                text_bg_coord2 = (xmin + text_size[0], ymin)
                cv2.rectangle(output_frame, text_bg_coord1, text_bg_coord2, dominant_color, -1)
                
                # Etiketi yazdır
                cv2.putText(output_frame, text, (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Son işlenen zaman güncelle
        last_processed_time = time.time()
        
        # İşlenmiş görüntüyü göster
        cv2.imshow('Video with Detected Objects and Colors', output_frame)
    
    # 'q' tuşuna basarak çıkabilirsin
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cv2.destroyAllWindows()