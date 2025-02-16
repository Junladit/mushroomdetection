import json
import os
import cv2
import glob
import time
from ultralytics import YOLO
import gspread
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials
from datetime import datetime
import sys

# ตั้งค่า console ให้รองรับ UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# ตั้งค่าการเชื่อมต่อกับ Google Sheets
SERVICE_ACCOUNT_FILE = "C:\Users\junladit\Desktop\yolotrain\myservice.json"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

try:
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(credentials)
    spreadsheet = gc.open('linebot')
    worksheet = spreadsheet.sheet1
    data_worksheet = spreadsheet.worksheet('data')
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Google Sheets: {e}")
    sys.exit(1)  # หยุดโปรแกรมถ้าเชื่อมต่อไม่ได้

# ตั้งค่าการเชื่อมต่อกับ Google Drive
drive_service = build('drive', 'v3', credentials=credentials)

# 📂 กำหนดโฟลเดอร์ต่าง ๆ
image_folder = "image"  # โฟลเดอร์เก็บภาพจากกล้อง
detected_folder = "detected_images"  # โฟลเดอร์เก็บภาพที่ตรวจจับแล้ว
training_folder = "training"  # โฟลเดอร์เก็บภาพที่ตรวจจับแล้วสำหรับ training
json_file_path = "output_data.json"
folder_id = "1i3hQizoo73C2e4_9p7_3c0Moc-mGNVNT"  # ใส่ Folder ID ของคุณใน Google Drive

# 📌 สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(image_folder, exist_ok=True)
os.makedirs(detected_folder, exist_ok=True)
os.makedirs(training_folder, exist_ok=True)

# 🚀 **โหลดโมเดล YOLO**
model = YOLO('C:/Users/junladit/Desktop/yolotrain/runs/detect/train4/weights/best.pt')

# 📸 **ถ่ายรูปจากกล้องแล้วเซฟลงโฟลเดอร์ `image/`**
def capture_image():
    cap = cv2.VideoCapture(0)  # เปิดกล้องตัวหลัก
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องได้")
        return None
    
    ret, frame = cap.read()
    cap.release()  # ปิดกล้อง

    if ret:
        # ตั้งชื่อไฟล์ เช่น "01.png", "02.png", ...
        image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
        new_image_number = len(image_files) + 1  # เลขภาพถัดไป
        image_path = os.path.join(image_folder, f"{new_image_number:02}.png")

        cv2.imwrite(image_path, frame)  # บันทึกรูป
        print(f"📷 ถ่ายรูปสำเร็จ: {image_path}")
        return image_path
    else:
        print("❌ ถ่ายรูปไม่สำเร็จ")
        return None

# 🖼 **ดึงไฟล์รูปล่าสุดจากโฟลเดอร์ `image/`**
def get_latest_image():
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    return image_files[-1] if image_files else None

# 📤 **อัปโหลดรูปภาพไปยัง Google Drive**
def upload_image_to_drive(image_path, folder_id):
    try:
        # สร้าง metadata สำหรับไฟล์
        file_metadata = {
            'name': os.path.basename(image_path),  # ชื่อไฟล์
            'parents': [folder_id]  # ID ของโฟลเดอร์ปลายทาง
        }

        # อัปโหลดไฟล์
        media = MediaFileUpload(image_path, resumable=True)
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name'
        ).execute()

        print(f"✅ อัปโหลดไฟล์สำเร็จ: {file.get('name')} (ID: {file.get('id')})")
        return True, file.get('name')
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดขณะอัปโหลดไฟล์: {e}")
        return False, None

# 🏁 **เริ่มทำงาน**
captured_image = capture_image()  # ถ่ายรูป
if captured_image:
    latest_image = get_latest_image()  # ดึงภาพล่าสุดจากโฟลเดอร์ image
    if latest_image:
        # 📌 ใช้ YOLO ตรวจจับวัตถุ
        results = model(latest_image)  # ตรวจจับวัตถุ

        # 🔍 ดึงข้อมูลผลลัพธ์
        labels = results[0].names  # รายชื่อคลาส
        detections = results[0].boxes  # กล่องที่ตรวจพบ

        # โหลดภาพเพื่อวาดผลลัพธ์
        image = cv2.imread(latest_image)

        # จัดรูปแบบข้อมูลก้อนใหม่
        detection_dict = {}
        for idx, box in enumerate(detections):
            class_id = int(box.cls.item())  # ดึงค่า class ID
            label = labels[class_id]  # แปลงเป็นชื่อ class
            detection_dict[f"ก้อนที่{idx+1}"] = label  # เช่น "ก้อนที่1": "big"

            # วาด bounding box และ label บนภาพ
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # ดึงค่าพิกัด bounding box
            confidence = float(box.conf[0])  # ดึงค่า confidence
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # วาดกล่อง
            cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # วาด label

        # 📂 โหลดข้อมูลเก่าจากไฟล์ JSON ถ้ามีอยู่แล้ว
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # เพิ่มข้อมูลใหม่
        existing_data.append(detection_dict)

        # 🔄 บันทึกลงไฟล์ JSON
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

        print(f"✅ บันทึกข้อมูลลง {json_file_path}")

        # 📂 บันทึกภาพที่มีการวาด bounding box และ label ลงในโฟลเดอร์ training
        image_filename = os.path.basename(latest_image)
        training_image_path = os.path.join(training_folder, image_filename)
        cv2.imwrite(training_image_path, image)  # บันทึกภาพที่วาดแล้ว
        print(f"✅ บันทึกภาพที่ตรวจจับแล้วไปยัง {training_image_path}")

        # อัปเดตข้อมูลใน Google Sheets
        try:
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)

            if not json_data:
                print("⚠ ไฟล์ JSON ว่างเปล่า ไม่สามารถอัปเดตข้อมูลได้")
            else:
                last_entry = json_data[-1]
                required_keys = [f"ก้อนที่{i}" for i in range(1, 6)]
                values = [last_entry.get(key, "ไม่พบข้อมูล") for key in required_keys]

                worksheet.batch_update([ 
                    {"range": "B1", "values": [[values[0]]]}, 
                    {"range": "B2", "values": [[values[1]]]}, 
                    {"range": "B3", "values": [[values[2]]]}, 
                    {"range": "B4", "values": [[values[3]]]}, 
                    {"range": "B5", "values": [[values[4]]]}, 
                    {"range": "B6", "values": [[values[4]]]}, 
                ])

                next_row_data = len(data_worksheet.col_values(1)) + 1
                data_worksheet.batch_update([ 
                    {"range": f"A{next_row_data}", "values": [[values[0]]]}, 
                    {"range": f"B{next_row_data}", "values": [[values[1]]]}, 
                    {"range": f"C{next_row_data}", "values": [[values[2]]]}, 
                    {"range": f"D{next_row_data}", "values": [[values[3]]]}, 
                    {"range": f"E{next_row_data}", "values": [[values[4]]]}, 
                    {"range": f"F{next_row_data}", "values": [[values[4]]]}, 
                ])

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"✅ อัปเดตข้อมูลสำเร็จ: {current_time}")

        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดขณะอัปเดตข้อมูล: {e}")

        # ดึงไฟล์ล่าสุดจากโฟลเดอร์ image
        latest_image = get_latest_image()
        if latest_image:
            # อัปโหลดรูปไปยัง Google Drive
            uploaded_file = upload_image_to_drive(latest_image, folder_id)
            if uploaded_file:
                print(f"✅ รูปภาพถูกอัปโหลดไปยัง Google Drive: {uploaded_file[1]}")
        else:
            print("❌ ไม่พบไฟล์รูปในโฟลเดอร์ `image/`")
    else:
        print("❌ ไม่พบไฟล์รูปในโฟลเดอร์ `image/`")

    # time.sleep(5)  # หยุด 5 วินาที