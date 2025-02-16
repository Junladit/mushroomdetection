import json
import os
import cv2
import glob
import time
import requests
import gspread
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials
from datetime import datetime
import sys

# ตั้งค่า console ให้รองรับ UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# ตั้งค่าการเชื่อมต่อกับ Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

try:
    creds = Credentials.from_service_account_file("myservice.json", scopes=SCOPES)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open('linebot')
    worksheet = spreadsheet.sheet1
    data_worksheet = spreadsheet.worksheet('data')
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Google Sheets: {e}")
    sys.exit(1)  # หยุดโปรแกรมถ้าเชื่อมต่อไม่ได้

# ตั้งค่าการเชื่อมต่อกับ Google Drive
drive_service = build('drive', 'v3', credentials=creds)

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

# 🚀 **ตั้งค่า Roboflow API**
ROBOFLOW_API_KEY = 'dyO5PlGICDRRTo6Wd6Q2'
ROBOFLOW_MODEL_ID = 'mushroom-709o6'
ROBOFLOW_VERSION = '1'
ROBOFLOW_API_URL = f'https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=20'

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
        # 📌 ใช้ Roboflow API ตรวจจับวัตถุ
        with open(latest_image, 'rb') as image_file:
            response = requests.post(ROBOFLOW_API_URL, files={'file': image_file})

        if response.status_code == 200:
            results = response.json()
            print(f"✅ ผลลัพธ์จาก Roboflow: {results}")

            # 🔍 ดึงข้อมูลผลลัพธ์
            detections = results['predictions']

            # โหลดภาพเพื่อวาดผลลัพธ์
            image = cv2.imread(latest_image)

            # จัดรูปแบบข้อมูลก้อนใหม่
            detection_dict = {}
            for idx, detection in enumerate(detections):
                label = detection['class']
                confidence = detection['confidence']
                detection_dict[f"ก้อนที่{idx+1}"] = label  # เช่น "ก้อนที่1": "big"

                # คำนวณตำแหน่ง bounding box จาก x, y, width, height
                x = int(detection['x'])
                y = int(detection['y'])
                width = int(detection['width'])
                height = int(detection['height'])

                x1 = int(x - width / 2)  # คำนวณ x1
                y1 = int(y - height / 2)  # คำนวณ y1
                x2 = int(x + width / 2)  # คำนวณ x2
                y2 = int(y + height / 2)  # คำนวณ y2

                # วาด bounding box และ label บนภาพ
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
            print(f"❌ เกิดข้อผิดพลาดขณะเรียกใช้ Roboflow API: {response.status_code}")
    else:
        print("❌ ไม่พบไฟล์รูปในโฟลเดอร์ `image/`")


#         {
#   "type": "service_account",
#   "project_id": "stunning-hull-447607-j5",
#   "private_key_id": "5522a5c1f09958aef58771c185b85963ad26f5bd",
#   "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCgRJVjzCLECpRW\nHHOJooTMmnbU7TDxhSyZKTh0WupJew7F53nzBKIgfdkcVYrhu+D00HaEad4G8bgB\n5fdxlewkBvcTqIAM1u4p4dKX0UdlePg1tXOBGwu5c7C9Ph/fn4OeI1TOXgfGnWnL\nwJDalBQwFhCOL2ornFRF3Rl8EI6SorOqXHWArGqS+5yKBG2oqOKWqsANq0XahMvs\nAAH+xHFFjn7JlQb3vaDkdhoba+eDSIaye7uZwF4+YlbTO3PgU0bDbZMsBllGm0U9\nwu9XMCCL9h7ycMsqNAgHEWvlgOpTLFNR1I4KEqlzs1PKZJR7Cm8uzw8gFGsNtodg\ndXG+MZJ1AgMBAAECggEAHiyrf82oYgan5QqYSjqaDDes1ewOgfqT6gZTxbx0Tf9o\nSaSKVlHyAHIRVX9ZlaSc9xrkpYuVLPOHtHvKuc0DV8kj7cSUz1YMI0CMON9DNPyw\nptQR+iXQcEsy5H5029KZokV+qxgTGLb8e0s3OqAUqPUOiuasc6eMSlcVfEFVxo+l\nPD8CcLmcSEilg+n4VFp3Rfaqm8sP2tYCM4HihSFEjiFqvLAxWHEqlE1czL7IjIXa\n5MOlvJneRZ8FnlL30AuMgN526wKcbiimQgZOa0IKDLoH1FUYKOij88tQxqt3RPVF\nk9If1Alshn69+hCI6cTkxlhsEXB9HQcbXZb5npdoIQKBgQDQYQLqm/yOdJjThNK2\nNUzMvWaQHcVB7kDn6PpVkgAgqkzEUKUUenv1/QUBXw/I3D+o+G6qOg8ZaThNwFaw\npeZIoBaWGLP22sE5SYSZLKa6CuzUGibb+0ScdF72rACQQdO8jTdlOSAT0xXC3ePb\n4ZVSjHuPq40vwa73xexVoOanTQKBgQDE5OL6iiUffLRI1RJCPisGoc9uReG/qqpR\npnI+Qb3T2FufqjFSLtzl26NINASKIzLJoW8BT29Q6wvif15hR8I7UBzr6F00kx/k\n2jGr7Gj3UxRcHqo7kxYTPjrGrbOZRpfLbCy3fdQ4OZ7nprgjK+/XVVmRd3Z6Jn5E\ntmRctaOTyQKBgQCBJO6yynQPMfIZfYM+C/CKH4Q2I01CnE+7qyei6vaLSCCFttlo\nxLSY8vQsMmdM/Du0FCw58fuzqwOLJH5VwqvjLNxyr+KxRkhkocy9RfAa83Rty7tz\nsNmIAZNtW5KJ1VJN1FOVt37K6pLdD7oNZ1StMYXOt+qrw2UWCKN9OlhZLQKBgBWY\nVRzNkgzGDLAATqRdVTLBBJM/rubqvQt/igAyDbPygvocHJS69xdu45XDvsu32JYs\n0pP+NmNVpFQPTDa3PCJtQv7M2Ywupsze8Zu9rjWSMyV3Z4xpMX6i0KeB3bTt/TAe\niTkG4APargcSThftdbzUa6J8y83R8v9uUcupUGuhAoGBAKdxAsR3TYfAUsvmrnT0\nEDvN0O4PI3ID38DSzQ/85INuLGSNbbRTvtRLJqugsOcidUofYW7XTR12MjfmVTXK\n8l3OSaRn1cExurYkKef7SR7SWyAVju3xiONu5f25TwybWgeUCIYQRtijN2y6fC7w\n21JOBbgmvEbdeRuGL6kfplS2\n-----END PRIVATE KEY-----\n",
#   "client_email": "my-service-account@stunning-hull-447607-j5.iam.gserviceaccount.com",
#   "client_id": "101389116624633606164",
#   "auth_uri": "https://accounts.google.com/o/oauth2/auth",
#   "token_uri": "https://oauth2.googleapis.com/token",
#   "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
#   "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/my-service-account%40stunning-hull-447607-j5.iam.gserviceaccount.com",
#   "universe_domain": "googleapis.com"
# }
