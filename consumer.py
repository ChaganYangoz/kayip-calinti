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
    'umbrella': 'Personal Item',
}

# Aktif filtre ayarları - başlangıçta tüm kişisel eşyalar aktif
active_filters = {
    'cell phone': True,
    'wallet': True,  # Cüzdan
    'handbag': True,  # Çanta
    'backpack': True,  # Sırt çantası
    'purse': True,  # El çantası
    'laptop': True,  # Dizüstü bilgisayar
}

# Klavye kısayolları için nesne-tuş eşleştirmesi
key_map = {
    '1': 'cell phone',  # 1 tuşu: Telefon filtresini aç/kapat
    '2': 'wallet',      # 2 tuşu: Cüzdan filtresini aç/kapat
    '3': 'handbag',     # 3 tuşu: Çanta filtresini aç/kapat
    '4': 'backpack',    # 4 tuşu: Sırt çantası filtresini aç/kapat
    '5': 'purse',       # 5 tuşu: El çantası filtresini aç/kapat
    '6': 'laptop',      # 6 tuşu: Dizüstü bilgisayar filtresini aç/kapat
    'a': 'all'          # a tuşu: Tüm nesneleri göster/gizle
}

# Renk isimlerine karar vermek için geliştirilmiş fonksiyon
def get_color_name(rgb):
    import numpy as np
    import cv2
    
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
    bgr_pixel = rgb  # Daha açık olmak için yeniden adlandırıyoruz
    
    # Giriş BGR'yi HSV'ye dönüştür
    bgr = np.uint8([[bgr_pixel]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    
    # HSV değerlerini al
    h, s, v = hsv
    
    # Siyah/beyaz/gri kontrolü - öncelikli kontrol
    # Daha kesin eşik değerleri
    if s < 30:  # Düşük doygunluk - gri tonları
        if v < 40:
            return "black"
        elif v > 220:
            return "white"
        else:
            return "gray"
    
    # Parlaklık çok düşükse siyah olarak değerlendir
    if v < 30:
        return "black"
    
    # Renk aralıkları (OpenCV'de Hue 0-179 arasında)
    # OpenCV'deki HSV değerleri: H: [0, 179], S: [0, 255], V: [0, 255]
    
    # Kırmızı (iki aralık - hem başlangıç hem bitiş)
    if (0 <= h <= 10) or (170 <= h <= 179):
        if s > 150 and v > 150:
            return "red"
        else:
            # Koyu kırmızı veya kahverengi
            return "brown" if v < 150 else "red"
    
    # Turuncu
    if 11 <= h <= 25:
        return "orange"
    
    # Sarı
    if 26 <= h <= 35:
        return "yellow"
    
    # Yeşil
    if 36 <= h <= 85:
        if v < 100:  # Koyu yeşil
            return "dark green" if "dark green" in colors else "green"
        else:
            return "green"
    
    # Açık mavi
    if 86 <= h <= 100:
        return "light blue"
    
    # Mavi
    if 101 <= h <= 125:
        if v < 120:  # Koyu mavi
            return "navy" if s > 100 else "dark blue"
        else:
            return "blue"
    
    # Mor
    if 126 <= h <= 155:
        return "purple"
    
    # Pembe
    if 156 <= h <= 169:
        return "pink"
    
    # Hiçbir özel durum yoksa, en yakın rengi bul
    min_distance = float('inf')
    closest_color = "unknown"
    
    for color_name, color_bgr in colors.items():
        # Her rengi HSV'ye dönüştür
        bgr_sample = np.uint8([[color_bgr]])
        hsv_sample = cv2.cvtColor(bgr_sample, cv2.COLOR_BGR2HSV)[0][0]
        
        # H, S ve V değerlerini karşılaştır, farklı ağırlıklarla
        h1, s1, v1 = map(int, hsv)
        h2, s2, v2 = map(int, hsv_sample)

        # Dairesel olan hue değerinde en kısa mesafeyi bul
        h_dist = min(abs(h1 - h2), 180 - abs(h1 - h2))
        
        # Hue'ye daha fazla ağırlık ver (renk tonu en önemli)
        distance = h_dist * 2.5 + abs(s1 - s2) * 0.8 + abs(v1 - v2) * 0.5
        
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    return closest_color

# Bir görüntüdeki baskın rengi bulma fonksiyonu
def get_dominant_color(image, k=5):
    import numpy as np
    import cv2
    from collections import Counter
    
    # Görüntüyü yumuşat - gürültü azaltma
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Görüntüyü küçült - daha hızlı işlem için
    height, width = image.shape[:2]
    if height > 300 or width > 300:
        scale = min(300 / width, 300 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        blurred = cv2.resize(blurred, (new_width, new_height))
    
    # HSV'ye dönüştür
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Gri tonlama oluştur
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Adaptif thresholding uygula - daha iyi nesne algılama
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 21, 5)
    
    # Morfolojik işlemler - gürültü azaltma
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Maskeyi uygula
    masked_image = cv2.bitwise_and(blurred, blurred, mask=mask)
    
    # Siyah olmayan pikselleri al
    non_black_pixels = masked_image[np.where((masked_image != [0,0,0]).all(axis=2))]
    
    # Eğer hiç piksel kalmadıysa veya çok az piksel varsa, orijinal görüntüyü kullan
    if len(non_black_pixels) < 100:
        # Orijinal görüntü üzerinde daha geniş bir analiz yap
        # Kenarları yumuşat
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)
        mask = 255 - edges  # Kenar olmayan bölgeleri al
        
        masked_image = cv2.bitwise_and(blurred, blurred, mask=mask)
        non_black_pixels = masked_image[np.where((masked_image != [0,0,0]).all(axis=2))]
        
        if len(non_black_pixels) < 100:
            # Hala yeterli piksel yoksa orijinali kullan
            pixels = blurred.reshape(-1, 3).astype(np.float32)
        else:
            pixels = non_black_pixels.astype(np.float32)
    else:
        pixels = non_black_pixels.astype(np.float32)
    
    # K-means algoritması için durma kriterleri (daha fazla iterasyon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    
    # K-means uygula - daha fazla küme ile başla ve sonra en önemlilerini seç
    k = min(k, len(pixels))
    if k == 0:
        return (0, 0, 0)  # Boş görüntü durumu
    
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 15, cv2.KMEANS_PP_CENTERS)
    
    # Kümeleri önem sırasına göre sırala
    counts = Counter(labels.flatten())
    
    # En çok bulunan 3 renk
    dominant_colors = []
    for i, (cluster_idx, count) in enumerate(counts.most_common(3)):
        if i >= len(counts):
            break
        color = tuple(map(int, centers[cluster_idx]))
        dominant_colors.append((color, count))
    
    # Arka plan olabilecek renkleri filtrele (genellikle beyaz veya siyah)
    filtered_colors = []
    for color, count in dominant_colors:
        bgr = np.uint8([[color]])
        hsv_color = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        
        # Düşük doygunluk veya çok düşük/yüksek parlaklık -> arka plan olabilir
        if hsv_color[1] > 30 or (30 < hsv_color[2] < 225):
            filtered_colors.append((color, count))
    
    # Filtreleme sonrası renk kalmadıysa, en büyük kümeden devam et
    if not filtered_colors and dominant_colors:
        return dominant_colors[0][0]
    elif filtered_colors:
        return filtered_colors[0][0]
    else:
        # Hiç renk bulunamazsa varsayılan değer
        return (128, 128, 128)  # Gri

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

from ultralytics import YOLO

model = None

try:
    model = YOLO('yolov8x.pt')
    print("YOLOv8x modeli yüklendi!")
except:
    try:
        model = YOLO('yolov8l.pt')
        print("YOLOv8l modeli yüklendi!")
    except:
        try:
            model = YOLO('yolov8m.pt')
            print("YOLOv8m modeli yüklendi!")
        except Exception as e:
            print("Hiçbir model yüklenemedi:", e)

# Modelin algılama parametrelerini ayarla - optimal tespit için
model.conf = 0.40  # Güven eşiği - kararlı tespit için orta seviye
model.iou = 0.45   # IOU eşiği - bu değer genellikle uygun
model.classes = None  # Tüm sınıfları tespit et
model.multi_label = True  # Çoklu etiket tespiti - örtüşen nesneler için faydalı
model.max_det = 50   # Makul sayıda tespit - performans/doğruluk dengesi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)


# Kafka consumer'ı başlat
consumer = KafkaConsumer(
    'topic_0',
    bootstrap_servers='pkc-4nmjv.francecentral.azure.confluent.cloud:9092',
    security_protocol='SASL_SSL',
    sasl_mechanism='PLAIN',
    sasl_plain_username='PEUKERKJNVCUFIPN',
    sasl_plain_password='nGptcmIG90ml6MIoAP4kYQ870NnctVV1lBMTre4WTrd9695wIomom1Lmrx0cPjAY',
    value_deserializer=lambda x: x.decode('utf-8'),
    auto_offset_reset='latest',
    max_partition_fetch_bytes=10485760
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

valid_classes = {'cell phone', 'wallet', 'handbag', 'backpack', 'purse', 'laptop'}

# Aktif filtreleri tutan sözlük (veya set)
active_filters = {cls: False for cls in valid_classes}

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
    print("Tüm nesneleri görmek için 'a' girin")
    
    try:
        # Kullanıcı inputunu al
        user_input = input("Filtreler: ").strip()
        
        # Tüm filtreleri kapat
        for filter_name in active_filters:
            active_filters[filter_name] = False
        
        if user_input.lower() == 'a':
            print("Tüm geçerli nesneler gösterilecek.")
            # Burada tüm YOLO nesnelerini aktifleştir
            for filter_name in active_filters:
                active_filters[filter_name] = True
        else:
            # Seçilen filtreleri aç
            choices = user_input.split()
            for choice in choices:
                if choice in key_map:
                    filter_name = key_map[choice]
                    active_filters[filter_name] = True
                    print(f"'{filter_name}' filtresi aktifleştirildi.")
            
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
                
                # Yolov8 model ile nesne tespiti
                results = model(enhanced_frame)  # Yolov8 modelini kullanarak tespit yap
                
                # Tespit edilen nesneler
                detected_items = []

                for result in results:
                    # Her tespit edilen nesne için
                    for box in result.boxes:
                        # Koordinatları ve etiket bilgilerini al
                        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                        label = result.names[int(box.cls)]  # Etiket adı
                        conf = box.conf[0].item()  # Güven skoru
                        
                        # Filtreleme mantığı
                        if label.lower() in active_filters and active_filters[label.lower()]:
                            detected_items.append(f"{label} ({conf:.2f})")
                            
                            # Nesnenin bulunduğu bölgeyi kes
                            roi = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                            
                            # Eğer ROI boş değilse
                            if roi.size > 0:
                                # Nesnenin baskın rengini bul
                                dominant_color = get_dominant_color(roi)
                                color_name = get_color_name(dominant_color)
                                
                                # Tespit edilen nesne etrafına kutu çiz
                                cv2.rectangle(enhanced_frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
                                
                                # Etiketi yazdır - güven skorunu da ekle
                                text = f"{label} ({color_name}) {conf:.2f}"
                                cv2.putText(enhanced_frame, text, (int(xmin), int(ymin)-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Tespit edilen nesneleri yazdır
                if detected_items:
                    print(f"Detected items: {', '.join(detected_items)}")
                
                processed_frames += 1
                
                # Son işlenen zaman güncelle
                last_processed_time = time.time()
                
                # İşlenmiş görüntüyü göster
                cv2.imshow('Video with Detected Objects and Colors', enhanced_frame)
            
            # Tuş yakalama
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False  # Çık
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

# Ana döngü - video yeniden oynatma seçeneği
replay = True
while replay:
    replay = process_video_stream()

# Kaynakları serbest bırak
consumer.close()
cv2.destroyAllWindows()
print("Program sonlandırıldı.")