import cv2
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime

# --- CONFIGURATION ---
TRAIN_PATH = 'Training_images'
ENCODINGS_FILE = 'encoded_data.pkl'
ATTENDANCE_FILE = 'Attendance.csv'
FACE_TOLERANCE = 0.45  # Lower is stricter
FRAME_SKIP_RATE = 2 

class AttendanceSystem:
    def __init__(self):
        self.known_encodings = []
        self.class_names = []
        self.marked_today = set()
        self.load_known_faces()
        self.load_today_attendance()

    def load_known_faces(self):
        """Loads encodings from pickle if available, otherwise generates them."""
        if os.path.exists(ENCODINGS_FILE):
            print("Loading existing encodings...")
            with open(ENCODINGS_FILE, 'rb') as f:
                self.known_encodings, self.class_names = pickle.load(f)
        else:
            print("No encoding file found. Generating new encodings...")
            images = []
            file_list = [f for f in os.listdir(TRAIN_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for file in file_list:
                img = cv2.imread(f'{TRAIN_PATH}/{file}')
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    self.class_names.append(os.path.splitext(file)[0].upper())
            
            self.known_encodings = [face_recognition.face_encodings(img)[0] for img in images if face_recognition.face_encodings(img)]
            
            with open(ENCODINGS_FILE, 'wb') as f:
                pickle.dump((self.known_encodings, self.class_names), f)
        
        print(f"System ready. {len(self.class_names)} faces loaded.")

    def load_today_attendance(self):
        """Syncs the 'marked_today' set with the CSV so we don't double-mark on restart."""
        if os.path.exists(ATTENDANCE_FILE):
            date_today = datetime.now().strftime("%d/%m/%Y")
            with open(ATTENDANCE_FILE, 'r') as f:
                for line in f:
                    entry = line.strip().split(',')
                    if len(entry) >= 2 and entry[1] == date_today:
                        self.marked_today.add(entry[0])

    def mark_attendance(self, name):
        """Logs attendance to CSV if not already logged today."""
        if name not in self.marked_today:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y,%H:%M:%S")
            with open(ATTENDANCE_FILE, 'a') as f:
                # Add header if file is empty
                if os.stat(ATTENDANCE_FILE).st_size == 0:
                    f.write('Name,Date,Time\n')
                f.write(f'{name},{dt_string}\n')
            
            self.marked_today.add(name)
            print(f"Log: {name} marked present.")

    def run(self):
        cap = cv2.VideoCapture(0)
        # Optimization: Lower resolution can significantly boost FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_counter = 0
        recent_locations = []
        recent_names = []

        while True:
            success, img = cap.read()
            if not success: break

            # 1. Process Frame (Optimization)
            if frame_counter % FRAME_SKIP_RATE == 0:
                # Small RGB image for fast processing
                small_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

                face_locs = face_recognition.face_locations(rgb_small)
                face_encodes = face_recognition.face_encodings(rgb_small, face_locs)

                recent_locations = []
                recent_names = []

                for encode, loc in zip(face_encodes, face_locs):
                    face_distances = face_recognition.face_distance(self.known_encodings, encode)
                    name = "UNKNOWN"

                    if len(face_distances) > 0:
                        best_match_idx = np.argmin(face_distances)
                        if face_distances[best_match_idx] < FACE_TOLERANCE:
                            name = self.class_names[best_match_idx]
                            self.mark_attendance(name)

                    recent_locations.append(loc)
                    recent_names.append(name)

            # 2. Draw Results
            for (top, right, bottom, left), name in zip(recent_locations, recent_names):
                # Scale back up (since we processed at 0.25x)
                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
                
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                cv2.rectangle(img, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Attendance System (Esc to Quit)', img)
            frame_counter += 1
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = AttendanceSystem()
    system.run()