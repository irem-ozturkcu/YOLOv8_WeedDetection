from ultralytics import YOLO

# YOLOv8 modelini başlat
model = YOLO('yolov8n.pt')  # YOLOv8 nano modelini kullanıyoruz

# Modeli eğit
model.train(
    data='data.yaml',  # Verinin bulunduğu YAML dosyası
    epochs=100,         # Eğitim süresi
    imgsz=640,         # Görüntü boyutu
    batch=16,          # Batch boyutu
    augment=True,      # Veri artırma etkin
    save=True          # En iyi ağırlıkları kaydet
)

# Eğitim sonrası doğrulama
metrics = model.val()
print(metrics)

