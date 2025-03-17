import os
from email.mime.text import MIMEText
import smtplib

def notifySelf(body):
    try:
        if os.path.exists('./Logs/voice.txt'):
            with open('./Logs/voice.txt', 'r') as file:
                subject = file.readline().strip()
                to_email = file.readline().strip()
                from_email = file.readline().strip()
                apiToken = file.readline().strip()
                msg = MIMEText(f'CS: {body}')
                msg["Subject"] = subject
                msg["From"] = from_email
                msg["To"] = to_email
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(from_email, apiToken)
                    server.sendmail(from_email, to_email, msg.as_string())
        else:
            print(body)
    except Exception as e:
        print("unable to send text...")
        print(e)



