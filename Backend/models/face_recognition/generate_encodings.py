# ==============================
# generate_encodings.py
# ==============================
# Run this script after adding new faces in faces/<PersonName>/ folder.
# It scans all folders and builds encodings.pkl used by criminal.py

import os
import cv2
import face_recognition
import pickle
from pathlib import Path

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "faces"
ENCODINGS_PATH = BASE_DIR / "encodings.pkl"

known_encodings = []
known_names = []

print("\n[INFO] Building face encodings database...")
print(f"[INFO] Faces directory: {FACES_DIR}\n")

# Iterate through each subfolder in "faces" (each person's name)
for person_name in os.listdir(FACES_DIR):
    person_folder = FACES_DIR / person_name
    if not os.path.isdir(person_folder):
        continue

    print(f"➡ Processing person: {person_name}")

    # Loop through all images in that folder
    for img_file in os.listdir(person_folder):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = person_folder / img_file
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        # Convert BGR (cv2) to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations
        boxes = face_recognition.face_locations(rgb_image, model="hog")
        if len(boxes) == 0:
            print(f"⚠️ No face found in {img_file}, skipping.")
            continue

        # Encode the face (take the first face found)
        encoding = face_recognition.face_encodings(rgb_image, boxes)[0]
        known_encodings.append(encoding)
        known_names.append(person_name)

    print(f"✅ Done: {person_name}")

# Save encodings
print("\n[INFO] Saving encodings to", ENCODINGS_PATH)
data = {"encodings": known_encodings, "names": known_names}
with open(str(ENCODINGS_PATH), "wb") as f:
    pickle.dump(data, f)

print(f"[SUCCESS] Encoded {len(known_names)} identities and saved to encodings.pkl")
