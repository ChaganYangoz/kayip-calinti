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
    'cup': 'Personal Item',
    'remote': 'Personal Item',
    'keyboard': 'Personal Item',
    'mouse': 'Personal Item',
    
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
    'couch': 'Furniture',
    'bed': 'Furniture',
    'table': 'Furniture',
    
    # Diğer
    'dog': 'Animal',
    'cat': 'Animal',
    'tv': 'Electronics',
    'tvmonitor': 'Electronics',
    'monitor': 'Electronics'
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
    'all': True  # Tüm nesneler - varsayılan olarak açık
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
    # OpenCV BGR formatında renk tanımları (BGR tupleları - NOT RGB!)
    # Format: (Blue, Green, Red)
    colors = {
        'red': (0, 0, 255),          # BGR = (0, 0, 255) → RGB = (255, 0, 0)
        'green': (0, 255, 0),        # BGR = (0, 255, 0) → RGB = (0, 255, 0)
        'blue': (255, 0, 0),         # BGR = (255, 0, 0) → RGB = (0, 0, 255)
        'dark blue': (139, 0, 0),    # BGR = (139, 0, 0) → RGB = (0, 0, 139)
        'light blue': (255, 191, 0), # BGR = (255, 191, 0) → RGB = (0, 191, 255)
        'yellow': (0, 255, 255),     # BGR = (0, 255, 255) → RGB = (255, 255, 0)
        'purple': (128, 0, 128),     # BGR = (128, 0, 128) → RGB = (128, 0, 128)
        'orange': (0, 165, 255),     # BGR = (0, 165, 255) → RGB = (255, 165, 0)
        'pink': (203, 192, 255),     # BGR = (203, 192, 255) → RGB = (255, 192, 203)
        'brown': (42, 42, 165),      # BGR = (42, 42, 165) → RGB = (165, 42, 42)
        'black': (0, 0, 0),          # BGR = (0, 0, 0) → RGB = (0, 0, 0)
        'white': (255, 255, 255),    # BGR = (255, 255, 255) → RGB = (255, 255, 255)
        'gray': (128, 128, 128),     # BGR = (128, 128, 128) → RGB = (128, 128, 128)
        'navy': (128, 0, 0)          # BGR = (128, 0, 0) → RGB = (0, 0, 128)
    }
    
    # Not: rgb parametresi gerçekte bir BGR değeri (OpenCV formatı)
    # O yüzden "rgb" yerine "bgr" olarak düşünülmeli
    bgr_pixel = rgb  # Daha açık olmak için yeniden adlandırıyoruz
    
    # RGB değişimi yapmaya gerek yok, çünkü zaten BGR formatında
    # En yakın rengi bul - HSV renk uzayında karşılaştır
    min_distance = float('inf')
    closest_color = "unknown"
    
    # Giriş BGR'yi HSV'ye dönüştür
    bgr = np.uint8([[bgr_pixel]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    
    # Hue değerini al (0-179)
    h, s, v = hsv
    
    # Koyu/açık renk kontrolü
    is_dark = v < 100  # Değer düşükse koyu
    low_sat = s < 50   # Doygunluk düşükse soluk
    
    # Siyah/beyaz/gri kontrolü
    if low_sat or (v < 50 and s < 100):
        if v < 50:
            return "black"
        elif v > 200 and s < 50:
            return "white"
        else:
            return "gray"
    
    # Özel renk durumları - Hue temel alınarak
    # Hue aralıkları (OpenCV'de 0-179 arasında)
    if 100 <= h <= 130:  # Mavi aralığı
        if is_dark or v < 150:
            return "dark blue" if v < 100 else "blue"
        else:
            return "light blue"
            
    if 0 <= h <= 10 or 170 <= h <= 180:  # Kırmızı aralığı
        return "red"
        
    if 20 <= h <= 30:  # Turuncu aralığı
        return "orange"
    
    # Standart karşılaştırma
    for color_name, color_bgr in colors.items():
        # Her rengi HSV'ye dönüştür
        bgr_sample = np.uint8([[color_bgr]])
        hsv_sample = cv2.cvtColor(bgr_sample, cv2.COLOR_BGR2HSV)[0][0]
        
        # Sadece H ve S'yi karşılaştır, V'yi dikkate alma
        h1, s1, _ = hsv
        h2, s2, _ = hsv_sample
        
        # Hue için dairesel mesafe (0-179 arasında)
        h_dist = min(abs(h1 - h2), 180 - abs(h1 - h2))
        
        # Ağırlıklı uzaklık hesapla (Hue daha önemli)
        distance = h_dist * 2 + abs(s1 - s2) * 0.8
        
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

# YOLOv5 modelini yükle - daha büyük ve hassas bir model (yolov5x)
try:
    # Önce daha büyük modeli yüklemeyi dene (yolov5x)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', force_reload=False, pretrained=True)
    print("YOLOv5x modeli yüklendi!")
except:
    try:
        # Eğer x modeli yüklenemezse l modeli dene
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l', force_reload=False, pretrained=True)
        print("YOLOv5l modeli yüklendi!")
    except:
        # Son çare olarak m modeli kullan
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=False, pretrained=True)
        print("YOLOv5m modeli yüklendi!")

# Modelin algılama parametrelerini ayarla - daha hassas tespit için
model.conf = 0.25  # Güven eşiğini düşür (0.45'ten 0.25'e)
model.iou = 0.45   # IOU eşiği
model.classes = None  # Tüm sınıfları tespit et
model.multi_label = True  # Çoklu etiket tespiti
model.max_det = 100  # Maksimum tespit sayısını artır

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
processed_frames = 0  # İşlenen kare sayısı

# Pencere boyutunu ayarla - daha büyük pencere
cv2.namedWindow('Video with Detected Objects and Colors', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video with Detected Objects and Colors', 1280, 720)

# Ekranda son durum bilgisini göster
def show_status_info(frame, processed_frames):
    height, width = frame.shape[:2]
    # Sol alt köşeye durum bilgisi ekle
    status_text = f"Processed frames: {processed_frames} | Press 'q' to quit"
    cv2.rectangle(frame, (10, height-40), (400, height-10), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (15, height-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame

print("Video işleme başlatılıyor. Video bittiğinde otomatik olarak sonlanacak veya 'q' tuşuna basarak çıkabilirsiniz.")

# Başlangıçta kullanıcıdan filtre inputu al
def get_filter_input():
    # Mevcut filtreleri göster
    print("\nFiltrelemek istediğiniz nesneleri seçin:")
    print("------------------------------------------")
    filter_options = []
    
    for key, value in key_map.items():
        print(f"{key}: {value}")
        filter_options.append(key)
    
    print("\nSeçmek istediğiniz filtreleri boşlukla ayırarak girin (örn: '1 2 3'):")
    print("Tüm nesneleri görmek için 'a' girin, sadece person için 'p' girin")
    
    try:
        # Kullanıcı inputunu al
        user_input = input("Filtreler: ").strip().split()
        
        # Tüm filtreleri kapat
        for filter_name in active_filters:
            active_filters[filter_name] = False
        
        # Seçilen filtreleri aç
        for choice in user_input:
            if choice in key_map:
                filter_name = key_map[choice]
                active_filters[filter_name] = True
                print(f"'{filter_name}' filtresi aktifleştirildi.")
        
        if 'a' in user_input:
            print("Tüm nesneler gösterilecek.")
        
        print("\nSeçtiğiniz filtreler ile video işleniyor...")
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        print("Varsayılan filtreler kullanılacak.")
    
    return active_filters

# Kullanıcıdan filtre girdisi al
get_filter_input()

# Video oynatma işlevi
def process_video_stream():
    global processed_frames, last_processed_time
    
    # Kafka'dan gelen veriyi işleme döngüsü
    for message in consumer:
        # Kafka'dan gelen base64 string verisini al
        frame_base64 = message.value

        try:
            # Video bitti mesajını kontrol et
            if frame_base64 == "END_OF_VIDEO":
                print("Video bittiği bildirimi alındı, işleme sonlandırılıyor...")
                print(f"Toplam {processed_frames} kare işlendi.")
                break
                
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

            # Her 0.2 saniyede bir frame işleme (daha sık işle)
            if elapsed_time >= 0.2:
                # Görüntüyü iyileştir - kontrast ve parlaklık ayarları
                enhanced_frame = frame.copy()
                enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=1.2, beta=10)  # Kontrast ve parlaklık artır
                
                # YOLOv5 ile nesne tespiti yap - geliştirilmiş parametrelerle
                results = model(enhanced_frame, size=640)  # Büyük bir boyutla tespit yap
                
                # Sonuçları pandas DataFrame olarak al
                df = results.pandas().xyxy[0]
                
                # Orijinal görüntünün bir kopyasını oluştur
                output_frame = enhanced_frame.copy()
                
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
                    
                    # Spesifik filtreleme mantığı
                    should_display = False
                    
                    # 1. Spesifik filtre kontrolü - örneğin 'person'
                    if label.lower() in active_filters:
                        # Bu nesne için spesifik filtre varsa, o filtre değerini kullan
                        should_display = active_filters[label.lower()]
                    # 2. Spesifik filtre yoksa ve 'all' aktifse göster
                    elif category is not None and active_filters.get('all', False):
                        should_display = True
                    # 3. Ne spesifik filtre ne de kategori varsa, 'all' aktifse göster
                    elif category is None and active_filters.get('all', False):
                        should_display = True
                    
                    # Gösterilmeyecekse atla
                    if not should_display:
                        continue
                    
                    # Nesneyi kaydet
                    detected_items.append(f"{label} ({conf:.2f})")
                    
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
                            'Animal': (255, 0, 255),       # Mor
                            'Electronics': (255, 128, 0)   # Açık mavi
                        }
                        
                        # Kategori varsa o renk, yoksa dominant rengi kullan
                        box_color = category_colors.get(category, dominant_color)
                        
                        # Tespit edilen nesne etrafına kutu çiz - kalınlığı artır
                        cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), box_color, 3)
                        
                        # Etiket metni oluştur - güven skorunu da ekle
                        if category:
                            text = f"{label} ({color_name}) {conf:.2f}"
                        else:
                            text = f"{label} ({color_name}) {conf:.2f}"
                        
                        # Metin için kutu çiz
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        text_bg_coord1 = (xmin, ymin - 25)
                        text_bg_coord2 = (xmin + text_size[0], ymin)
                        cv2.rectangle(output_frame, text_bg_coord1, text_bg_coord2, box_color, -1)
                        
                        # Etiketi yazdır - yazı boyutunu artır
                        cv2.putText(output_frame, text, (xmin, ymin - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Tespit sonuçlarını göster
                if detected_items:
                    print(f"Detected items: {', '.join(detected_items)}")
                
                # Aktif filtreleri ekranda gösterme (kaldırıldı)
                # output_frame = show_active_filters(output_frame)
                
                # Durum bilgisini ekranda gösterme (kaldırıldı)
                processed_frames += 1
                # output_frame = show_status_info(output_frame, processed_frames)
                
                # Son işlenen zaman güncelle
                last_processed_time = time.time()
                
                # İşlenmiş görüntüyü göster
                cv2.imshow('Video with Detected Objects and Colors', output_frame)
            
            # Tuş yakalama
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' tuşuna basarak çıkabilirsin
            if key == ord('q'):
                return False
                
            # Tuşa basıldıysa filtreleri güncelle
            key_pressed = chr(key) if key < 128 else ''
            
            if key_pressed in key_map:
                filter_name = key_map[key_pressed]
                active_filters[filter_name] = not active_filters[filter_name]
                print(f"Filter '{filter_name}' is now {'ON' if active_filters[filter_name] else 'OFF'}")
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
            
    # Video bitti, yeni bir ekran göster
    end_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(end_frame, "Video Processing Complete", (400, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(end_frame, f"Processed {processed_frames} frames", (450, 370), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(end_frame, "Press 'r' to replay or 'q' to quit", (430, 420), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Video with Detected Objects and Colors', end_frame)
    
    # Kullanıcı girişini bekle
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            return False  # Çık
        elif key == ord('r'):
            # Sayaçları sıfırla ve yeniden başlat
            processed_frames = 0
            last_processed_time = time.time()
            return True  # Yeniden oynat
            
    return False

# Ana döngü - video yeniden oynatma seçeneği
replay = True
while replay:
    replay = process_video_stream()

# Kaynakları serbest bırak
consumer.close()
cv2.destroyAllWindows()
print("Program sonlandırıldı.")