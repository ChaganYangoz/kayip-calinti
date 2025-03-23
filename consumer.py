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

# İlgilenilen nesne listesi (YOLO nesne sınıfları)
OBJECTS_OF_INTEREST = {
    # Kişisel eşyalar
    'cell phone': 'Personal Item',
    'phone': 'Personal Item',
    'mobile phone': 'Personal Item',
    'smartphone': 'Personal Item',
    'backpack': 'Personal Item',
    'handbag': 'Personal Item',
    'wallet': 'Personal Item',
    'purse': 'Personal Item',
    'suitcase': 'Personal Item',
    'laptop': 'Personal Item',
    'book': 'Personal Item',
    'umbrella': 'Personal Item',
    'bottle': 'Personal Item',
    
    # İnsanlar
    'person': 'Person',
    
    # Araçlar
    'bicycle': 'Vehicle',
    'car': 'Vehicle',
    'motorcycle': 'Vehicle',
    'bus': 'Vehicle',
    'truck': 'Vehicle',
    
    # Mobilyalar
    'chair': 'Furniture',
    'bench': 'Furniture',
    'dining table': 'Furniture',
    'desk': 'Furniture',
    
    # Diğer
    'dog': 'Animal',
    'cat': 'Animal'
}

# Aktif filtre ayarları - başlangıçta tüm kişisel eşyalar aktif
active_filters = {
    'cell phone': True,
    'wallet': True,  # Cüzdan
    'handbag': True,  # Çanta
    'backpack': True,  # Sırt çantası
    'purse': True,  # El çantası
    'laptop': True,  # Dizüstü bilgisayar
    'person': False,  # Kişi
    'all': False  # Tüm nesneler
}

# Klavye kısayolları için nesne-tuş eşleştirmesi
key_map = {
    '1': 'cell phone',  # 1 tuşu: Telefon filtresini aç/kapat
    '2': 'wallet',      # 2 tuşu: Cüzdan filtresini aç/kapat
    '3': 'handbag',     # 3 tuşu: Çanta filtresini aç/kapat
    '4': 'backpack',    # 4 tuşu: Sırt çantası filtresini aç/kapat
    '5': 'purse',       # 5 tuşu: El çantası filtresini aç/kapat
    '6': 'laptop',      # 6 tuşu: Dizüstü bilgisayar filtresini aç/kapat
    'p': 'person',      # p tuşu: Kişi filtresini aç/kapat
    'a': 'all'          # a tuşu: Tüm nesneleri göster/gizle
}

# Renk isimlerine karar vermek için fonksiyon
def get_color_name(rgb):
    # Yaygın renk isimlerini ve BGR değerlerini tanımla (OpenCV BGR formatında çalışır)
    colors = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),   # BGR formatında sarı
        'purple': (128, 0, 128),
        'orange': (0, 165, 255), # BGR formatında turuncu
        'pink': (203, 192, 255),
        'brown': (42, 42, 165),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'gray': (128, 128, 128)
    }
    
    # RGB değişimi yapmaya gerek yok, çünkü zaten BGR formatında
    # En yakın rengi bul - renk uzayında Öklid mesafesi kullanarak
    min_distance = float('inf')
    closest_color = "unknown"
    
    for color_name, color_bgr in colors.items():
        # HSV'ye dönüştürerek renk karşılaştırması daha doğru olabilir
        bgr1 = np.uint8([[rgb]])
        bgr2 = np.uint8([[color_bgr]])
        
        hsv1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2HSV)[0][0]
        hsv2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)[0][0]
        
        # Sadece renk tonu (hue) ve doygunluk (saturation) karşılaştır
        # Parlaklık (value) göz ardı edilebilir
        distance = np.sqrt((hsv1[0] - hsv2[0])**2 + (hsv1[1] - hsv2[1])**2) 
        
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    return closest_color

# Bir görüntüdeki baskın rengi bulma fonksiyonu
def get_dominant_color(image, k=5):
    # Maskeleme ile arka planı kaldır
    # Sadece önemli renkli pikselleri kullan
    
    # Gri tonlama oluştur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Otsu thresholding ile maske oluştur (nesne daha belirgin)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Maskeyi uygula
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Siyah olmayan pikselleri al
    non_black_pixels = masked_image[np.where((masked_image != [0,0,0]).all(axis=2))]
    
    # Eğer hiç piksel kalmadıysa, orijinal görüntüyü kullan
    if len(non_black_pixels) == 0:
        pixels = image.reshape(-1, 3).astype(np.float32)
    else:
        pixels = non_black_pixels.astype(np.float32)
    
    # K-means algoritması için durma kriterleri
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # K-means uygula
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # En çok kullanılan renk merkezi
    counts = Counter(labels.flatten())
    dominant_color = centers[counts.most_common(1)[0][0]]
    
    # BGR formatında döndür
    return tuple(map(int, dominant_color))

# Ekrandaki aktif filtreleri göster
def show_active_filters(frame):
    y_pos = 30
    cv2.putText(frame, "Active Filters:", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    y_pos += 25
    
    for item, is_active in active_filters.items():
        status = "ON" if is_active else "OFF"
        color = (0, 255, 0) if is_active else (0, 0, 255)  # Yeşil/Kırmızı
        
        # Klavye kısayolu bul
        key = [k for k, v in key_map.items() if v == item]
        shortcut = key[0] if key else ""
        
        text = f"{item} [{shortcut}]: {status}"
        cv2.putText(frame, text, (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_pos += 25
    
    # Yardım metni
    cv2.putText(frame, "Press number keys (1-6) to toggle filters, 'a' for all, 'q' to quit", 
                (10, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

# PyTorch modelini yükleme öncesi import etmek için
from torch import amp

# YOLOv5 modelini yükle
# Orijinal torch.hub.load kullanımı yerine doğrudan repo URL'si ile yükleme
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=False)  

# CPU'da çalışacak şekilde modeli ayarla
model.cpu()

# Kafka consumer'ı başlat
consumer = KafkaConsumer(
    'my-topic',  # Kafka topic adı
    bootstrap_servers=['localhost:29092'],
    value_deserializer=lambda x: x.decode('utf-8'),  # Base64 string olarak deserialize et
    auto_offset_reset='latest',  # En son mesajdan itibaren oku
    max_partition_fetch_bytes=10485760  # Buffer'ı artır (10MB)
)

# Son işlenen zaman (frame)
last_processed_time = time.time()
frame_buffer = []  # Buffer için frame listesi

# Kafka'dan gelen veriyi işleme döngüsü
for message in consumer:
    # Kafka'dan gelen base64 string verisini al
    frame_base64 = message.value

    try:
        # Base64 verisini çöz
        frame_bytes = base64.b64decode(frame_base64)

        # Çözülmüş byte verisini NumPy array'e çevir
        np_array = np.frombuffer(frame_bytes, dtype=np.uint8)

        # NumPy array'ini OpenCV formatına dönüştür
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        # Frame geçerli değilse atla
        if frame is None or frame.size == 0:
            continue

        # Geçen zaman
        elapsed_time = time.time() - last_processed_time

        # Her 0.3 saniyede bir frame işleme (daha sık işle)
        if elapsed_time >= 0.3:
            # YOLOv5 ile nesne tespiti yap
            results = model(frame)  # Görüntüyü YOLOv5 modeline ver
            
            # Sonuçları pandas DataFrame olarak al
            df = results.pandas().xyxy[0]
            
            # Orijinal görüntünün bir kopyasını oluştur
            output_frame = frame.copy()
            
            # Tespit edilen nesneleri filtrele
            detected_items = []
            
            # Her tespit edilen nesne için
            for i, row in df.iterrows():
                # Koordinatları ve etiketi al
                xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = row['name']
                conf = row['confidence']
                
                # Sadece ilgilenilen nesneleri işaretle
                category = OBJECTS_OF_INTEREST.get(label.lower(), None)
                
                # Filtre kontrolü - Eğer "all" filtresi aktifse veya nesne spesifik olarak filtrelendiyse
                should_display = active_filters.get('all', False)
                
                # Label filtreleme listesinde varsa ve aktifse göster
                if label.lower() in active_filters and active_filters[label.lower()]:
                    should_display = True
                
                # Eğer nesne filtreler arasında yoksa ve "all" filtresi aktif değilse gizle
                if not should_display and category is None:
                    continue
                
                # Nesneyi kaydet
                detected_items.append(label)
                
                # Nesnenin bulunduğu bölgeyi kes
                roi = frame[ymin:ymax, xmin:xmax]
                
                # Eğer ROI boş değilse
                if roi.size > 0:
                    # Nesnenin baskın rengini bul
                    dominant_color = get_dominant_color(roi)
                    color_name = get_color_name(dominant_color)
                    
                    # Kategori renklerini belirle
                    category_colors = {
                        'Personal Item': (0, 0, 255),  # Kırmızı (BGR)
                        'Person': (0, 255, 0),         # Yeşil
                        'Vehicle': (255, 0, 0),        # Mavi
                        'Furniture': (0, 255, 255),    # Sarı
                        'Animal': (255, 0, 255)        # Mor
                    }
                    
                    # Kategori varsa o renk, yoksa dominant rengi kullan
                    box_color = category_colors.get(category, dominant_color)
                    
                    # Tespit edilen nesne etrafına kutu çiz
                    cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), box_color, 2)
                    
                    # Etiket metni oluştur
                    if category:
                        text = f"{label} ({color_name}) - {category}"
                    else:
                        text = f"{label} ({color_name})"
                    
                    # Metin için kutu çiz
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_bg_coord1 = (xmin, ymin - 20)
                    text_bg_coord2 = (xmin + text_size[0], ymin)
                    cv2.rectangle(output_frame, text_bg_coord1, text_bg_coord2, box_color, -1)
                    
                    # Etiketi yazdır
                    cv2.putText(output_frame, text, (xmin, ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Tespit sonuçlarını göster
            if detected_items:
                print(f"Detected items: {', '.join(detected_items)}")
            
            # Aktif filtreleri ekranda göster
            output_frame = show_active_filters(output_frame)
            
            # Son işlenen zaman güncelle
            last_processed_time = time.time()
            
            # İşlenmiş görüntüyü göster
            cv2.imshow('Video with Detected Objects and Colors', output_frame)
        
        # Tuş yakalama
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' tuşuna basarak çıkabilirsin
        if key == ord('q'):
            break
            
        # Tuşa basıldıysa filtreleri güncelle
        key_pressed = chr(key) if key < 128 else ''
        
        if key_pressed in key_map:
            filter_name = key_map[key_pressed]
            active_filters[filter_name] = not active_filters[filter_name]
            print(f"Filter '{filter_name}' is now {'ON' if active_filters[filter_name] else 'OFF'}")
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        continue

# Kaynakları serbest bırak
consumer.close()
cv2.destroyAllWindows()