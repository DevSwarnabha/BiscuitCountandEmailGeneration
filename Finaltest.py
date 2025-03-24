import requests
import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
import csv
import datetime
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

model = YOLO(r'C:\Users\Bhai\Desktop\Computer_Vision_Intern\Biscuits\runs\detect\train5\weights\best.pt')

class_counts = {"Gold": 0, "Top": 0}

# url = "http://192.168.29.232:8080/shot.jpg"
url = "http://172.30.1.71:8080/shot.jpg"

window_width = 1000
window_height = 1400
cv2.namedWindow("Biscuit Predictor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Biscuit Predictor", window_width, window_height)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
bars = ax.bar(class_counts.keys(), class_counts.values(), color=["tab:blue", "tab:red"])
ax.set_xlabel('Biscuits')
ax.set_ylabel('Count')
ax.set_title('Count of Biscuits')
fig.tight_layout()
ax.set_ylim(0, 10)

file_name = "BiscuitCounts.csv"
try:
    with open(file_name, "r"):
        pass  # File exists, do nothing
except FileNotFoundError:
    with open(file_name, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Date", "Time Stamp", "Top", "Gold", "Low Brightness", "Idle"])

t1 = time.time()

low_brightness_text = None
idle_time = 0
idle_counter = 0

low_brightness_time = 0
low_brightness_counter = 0

while True:
    try:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)

        results = model.predict(source=frame, verbose=False)
        annotated_frame = results[0].plot()

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        if np.mean(gray_image) > 40:

            class_counts = {"Gold": 0, "Top": 0}
            for box in results[0].boxes:
                cls = int(box.cls)
                if cls == 0:
                    class_counts["Gold"] += 1
                elif cls == 1:
                    class_counts["Top"] += 1

            for i, rect in enumerate(bars):
                rect.set_height(class_counts[list(class_counts.keys())[i]])

            if low_brightness_text:
                low_brightness_text.remove()
                low_brightness_text = None

                

            if (class_counts["Gold"] + class_counts["Top"]) <= 2:
                idle_time += time.time() - idle_counter if idle_counter != 0 else 0
                idle_counter = time.time()

        else:

            if not low_brightness_text:
                low_brightness_text = ax.text(0.5, 0.5, 'Low Brightness', fontsize=30, color='red', ha='center', va='center', transform=ax.transAxes)
        
            low_brightness_time += time.time() - low_brightness_counter if low_brightness_counter != 0 else 0
            low_brightness_counter = time.time()
        
        idle_counter = time.time()
        low_brightness_counter = time.time()
        print(str(idle_time // 3600) + " : " + str(idle_time // 60) + " : " + str(round(idle_time % 60, 2)),"|||||", str(low_brightness_time // 3600) + " : " + str(low_brightness_time // 60) + " : " + str(round(low_brightness_time % 60, 2)))

        fig.canvas.draw()
        fig.canvas.flush_events()

        cv2.imshow("Biscuit Predictor", annotated_frame)

        t2 = time.time()
        if t2 - t1 >= 5:
            idle_time_str = str(idle_time // 3600) + " : " + str(idle_time // 60) + " : " + str(round(idle_time % 60, 2))
            low_brightness_time_str = str(low_brightness_time // 3600) + " : " + str(low_brightness_time // 60) + " : " + str(round(low_brightness_time % 60, 2))
            with open(file_name, "a", newline="") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([ 
                    datetime.datetime.now().strftime("%Y-%m-%d"),
                    datetime.datetime.now().strftime("%H:%M:%S"),
                    class_counts["Top"],
                    class_counts["Gold"],
                    low_brightness_time_str,
                    idle_time_str
                ])
            t1 = time.time()

        if cv2.waitKey(30) == 27:
            plt.close('all')
            cv2.destroyAllWindows()

            sender_email = "swarnabha.mitra04@gmail.com"
            receiver_email = "mitra.swarnabha04@gmail.com"
            password = input("Enter password: ")  

            # Email message
            subject = "Report for " + str(datetime.datetime.now().date())
            body = "This is an automated generated mail"

            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))

            # Attach the CSV file
            try:
                with open(file_name, "rb") as file:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(file.read())  # Load file content
                    encoders.encode_base64(part)  # Encode in base64
                    part.add_header("Content-Disposition", f"attachment; filename={file_name}")
                    message.attach(part)

                # Send the email
                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()  # Secure the connection
                    server.login(sender_email, password)
                    server.sendmail(sender_email, receiver_email, message.as_string())
                print("Email sent successfully!")
            except FileNotFoundError:
                print(f"Error: The file {file_name} does not exist.")
            except smtplib.SMTPException as e:
                print(f"SMTP error: {e}")
            except Exception as e:
                print(f"Unexpected error while sending email: {e}")

            break

    except requests.RequestException as e:
        print(f"Error fetching frame: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Cleanup resources
plt.ioff()
plt.show()
exit()
