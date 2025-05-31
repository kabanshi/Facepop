import cv2
import os
import sys
import sqlite3
from datetime import datetime
import numpy as np
import tkinter as tk
from tkinter import Label, Button, ttk
from PIL import Image, ImageTk
import face_recognition

class FaceApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Распознавание лиц")

        os.makedirs("saved_faces", exist_ok=True)
        os.makedirs("known_faces", exist_ok=True)

        self.init_db()
        self.load_known_faces()

        self.label = Label(window)
        self.label.pack()

        #Камера не открыта до нажатия "Старт"
        self.video = None
        self.camera_index = tk.IntVar(value=0)

        #Виджет выбора камеры
        tk.Label(window, text="Выбор камеры:").pack()
        self.camera_list = self.detect_cameras()
        self.camera_select = ttk.Combobox(window, values=self.camera_list, state="readonly")
        self.camera_select.current(0)
        self.camera_select.pack()

        #Пути к модели
        model_dir = getattr(sys, '_MEIPASS', os.path.abspath("."))
        prototxt_path = os.path.join(model_dir, "models", "deploy.prototxt")
        caffemodel_path = os.path.join(model_dir, "models", "res10_300x300_ssd_iter_140000.caffemodel")
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        #Кнопки
        self.btn_start = Button(window, text="Старт", command=self.start_video)
        self.btn_start.pack(side="left", padx=10)

        self.btn_stop = Button(window, text="Стоп", command=self.stop_video)
        self.btn_stop.pack(side="right", padx=10)

        self.running = False

    def init_db(self):
        self.conn = sqlite3.connect("faces.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS recognized_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                timestamp TEXT,
                image_path TEXT
            )
        """)
        self.conn.commit()

    def save_face_to_db(self, name, image_path):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("INSERT INTO recognized_faces (name, timestamp, image_path) VALUES (?, ?, ?)",
                            (name, timestamp, image_path))
        self.conn.commit()

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []

        for filename in os.listdir("known_faces"):
            if filename.lower().endswith((".jpg", ".png")):
                path = os.path.join("known_faces", filename)
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    self.known_face_names.append(name)

    def detect_cameras(self, max_tested=5):
        available = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                available.append(f"Камера {i}")
                cap.release()
        if not available:
            available.append("Нет доступных камер")
        return available

    def start_video(self):
        if self.running:
            return

        cam_text = self.camera_select.get()
        if not cam_text.startswith("Камера"):
            print("Камера не выбрана.")
            return

        index = int(cam_text.split()[-1])
        self.video = cv2.VideoCapture(index)

        if not self.video.isOpened():
            print("Не удалось открыть камеру.")
            return

        self.running = True
        self.update_frame()

    def stop_video(self):
        self.running = False
        if self.video and self.video.isOpened():
            self.video.release()

    def update_frame(self):
        if not self.running or not self.video:
            return

        ret, frame = self.video.read()
        if not ret:
            return

        rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                matched_idx = matches.index(True)
                name = self.known_face_names[matched_idx]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            face_crop = frame[top:bottom, left:right]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"saved_faces/{name}_{timestamp}.jpg"
            cv2.imwrite(filename, face_crop)
            self.save_face_to_db(name, filename)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        self.window.after(10, self.update_frame)

    def __del__(self):
        if self.video and self.video.isOpened():
            self.video.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()





