import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


class LettuceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marul Tespiti Uygulaması")
        self.root.geometry("500x1000")  # Arayüz boyutu
        self.root.resizable(False, False)

        # Arka plan resmi
        self.background_image = Image.open("background3.jfif")  # Arka plan için resim
        self.background_image = self.background_image.resize((500, 1000), Image.Resampling.LANCZOS)  # Boyutlandırma
        self.bg_image = ImageTk.PhotoImage(self.background_image)

        # Canvas ile arka planı ekleme
        self.canvas = tk.Canvas(self.root, width=500, height=1000)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.bg_image, anchor="nw")

        # YOLOv8 modelini yükleme
        self.model = YOLO('best.onnx')  # model

        # UI bileşenleri
        self.setup_ui()

        self.cap = None  # Video yakalayıcı
        self.running = False  # Akışı kontrol etme

    def setup_ui(self):
        # Düğme stilleri
        button_style = {
            'font': ('Arial', 14, 'bold'),
            'bg': '#D0EAC3',
            'activebackground': '#B2E4A1',
            'borderwidth': 2,
            'relief': 'solid',  # Düğme kenarları yumuşak yapılıyor
            'highlightthickness': 0,  # Highlight kenarlarını kaldır
            'padx': 10,
            'pady': 5
        }

        # Düğmeleri ortalamak için place kullanıyoruz (x ve y koordinatları ile)
        self.upload_button = tk.Button(self.root, text="Görüntü Yükle", command=self.load_image, **button_style)
        self.upload_button.place(relx=0.5, rely=0.6, anchor="center")

        self.video_button = tk.Button(self.root, text="Video Yükle", command=self.load_video, **button_style)
        self.video_button.place(relx=0.5, rely=0.65, anchor="center")

        self.camera_button = tk.Button(self.root, text="Kamerayı Başlat", command=self.start_camera, **button_style)
        self.camera_button.place(relx=0.5, rely=0.7, anchor="center")

        self.stop_button = tk.Button(self.root, text="Durdur", command=self.stop_stream, **button_style)
        self.stop_button.place(relx=0.5, rely=0.75, anchor="center")

        # Durum çubuğu
        self.status_bar = tk.Label(self.root, text="Hoşgeldiniz!", font=('Arial', 12), bg="#A8E6CE")
        self.status_bar.place(relx=0.5, rely=0.85, anchor="center")

        self.image_label = tk.Label(self.root, bg="#A8E6CE")
        self.image_label.place(relx=0.5, rely=0.2, anchor="center")

        self.result_label = tk.Label(self.root, text="", font=('Arial', 16), bg="#A8E6CE")
        self.result_label.place(relx=0.5, rely=0.45, anchor="center")

    def load_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            image = cv2.imread(self.file_path)
            self.process_and_display(image)

    def load_video(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.cap = cv2.VideoCapture(self.file_path)
            self.running = True
            self.status_bar.config(text="Video oynatılıyor...")
            self.process_video()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)  # Varsayılan kamera
        self.running = True
        self.status_bar.config(text="Kamera açıldı...")
        self.process_video()

    def stop_stream(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.image_label.config(image='')
            self.result_label.config(text="")
            self.status_bar.config(text="Akış durduruldu.")

    def process_and_display(self, img):
        results = self.model.predict(img)  # Tahmin yap

        predictions = []  # Tahminleri saklayacağız
        for result in results:  # Tüm tespitleri döngü ile kontrol et
            for *xyxy, conf, cls in result.boxes.data.tolist():  # Koordinatları al
                x1, y1, x2, y2 = map(int, xyxy)  # Koordinatları tam sayıya çevir
                label = f"{self.model.names[int(cls)]} {conf:.2f}"  # Etiketi oluştur
                predictions.append(label)  # Tahminleri sakla
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Çerçeve çiz
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Etiket yaz

        # Tahminleri arayüze yazdır
        self.result_label.config(
            text="Tahminler: " + ", ".join(predictions) if predictions else "Hiçbir nesne bulunamadı.")

        self.display_image(img)  # Görüntüyü güncelle

    def display_image(self, img):
        img = cv2.resize(img, (400, 300))  # Görüntüyü boyutlandır
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB formatına çevir
        img = np.array(img)
        img = cv2.imencode('.png', img)[1].tobytes()  # PNG formatına çevir
        img = tk.PhotoImage(data=img)
        self.image_label.config(image=img)
        self.image_label.image = img  # Referansı sakla

    def process_video(self):
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.process_and_display(frame)
                self.root.after(10, self.process_video)  # 10 ms bekle ve tekrar oku
            else:
                self.stop_stream()


if __name__ == "__main__":
    root = tk.Tk()
    app = LettuceDetectionApp(root)
    root.mainloop()
