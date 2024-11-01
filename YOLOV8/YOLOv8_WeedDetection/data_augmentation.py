import os
import cv2
import albumentations as A

# Klasör yolları
train_dir = 'datasets/images/train'  # Eğitim resimlerinin olduğu klasör
label_dir = 'datasets/labels/train'  # Etiket dosyalarının olduğu klasör
output_dir = 'datasets/images/train_augment'  # Artırılmış resimlerin kaydedileceği klasör

# Klasörü oluştur
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Veri artırma işlemleri
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Yatay çevirme
    A.VerticalFlip(p=0.5),  # Dikey çevirme
    A.Rotate(limit=25, p=0.5),  # Dönme
    A.GaussNoise(var_limit=(0, 25), p=0.5),  # Gürültü ekleme
    A.RandomBrightnessContrast(p=0.5),  # Parlaklık ve kontrast ayarlama
])

# Resim ve etiket dosyalarını işle
for filename in os.listdir(train_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Resim dosyasını oku
        img_path = os.path.join(train_dir, filename)
        image = cv2.imread(img_path)

        # Etiket dosyasını oku
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        with open(label_path, 'r') as file:
            labels = file.readlines()

        # Veri artırma işlemini uygula
        for i in range(5):  # 5 farklı artırılmış görüntü oluştur
            augmented_image = transform(image=image)['image']

            # Farklı isimlerle kaydet
            augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i + 1}.jpg"
            cv2.imwrite(os.path.join(output_dir, augmented_filename), augmented_image)

            # Her artırılmış görüntü için yeni bir etiket dosyası oluştur
            augmented_label_filename = f"{os.path.splitext(label_filename)[0]}_aug_{i + 1}.txt"
            augmented_label_path = os.path.join(output_dir, augmented_label_filename)

            # Orijinal etiketleri artırılmış görüntü için yaz
            with open(augmented_label_path, 'w') as augmented_file:
                # Orijinal etiketleri aynen kopyala
                augmented_file.writelines(labels)

print("Veri artırma işlemi tamamlandı!")
