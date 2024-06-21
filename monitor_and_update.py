
import os
import time
import pandas as pd
import joblib
from collections import deque
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(filename='logs/monitoring_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Directory to monitor for new data
directory_to_watch = 'new_data/'

# Load the saved model and memory buffer
sgd_pipeline = joblib.load('models/sgd_pipeline_weighted_memory.joblib')
scalable_memory_buffer = joblib.load('models/weighted_memory_buffer.joblib')

# Load the TF-IDF transformer
tfidf_transformer = sgd_pipeline.named_steps['tfidf']

# Function to process and update model with new data
def process_new_data(new_files):
    try:
        for file in new_files:
            new_data_df = pd.read_csv(os.path.join(directory_to_watch, file))
            X_new = new_data_df['Description']
            y_new = new_data_df['Step']
            auto_update_model(X_new, y_new)
            logging.info(f"Model updated with {file}")
    except Exception as e:
        logging.error(f"Error processing new data: {e}")
        notify_on_error(f"Error processing new data: {e}")

# Function to automatically update model with new data
def auto_update_model(new_data, new_labels):
    try:
        for desc, step in zip(new_data, new_labels):
            X_transformed = tfidf_transformer.transform([desc])
            sgd_pipeline.named_steps['sgd'].partial_fit(X_transformed, [step], classes=list(sgd_pipeline.named_steps['sgd'].classes_))
            scalable_memory_buffer.add(desc, step)
            log_update(desc, step)
    except Exception as e:
        logging.error(f"Error updating model: {e}")
        notify_on_error(f"Error updating model: {e}")

# Logging mechanism
def log_update(description, step):
    try:
        with open('logs/update_log.txt', 'a') as log_file:
            log_file.write(f"Updated with description: {description}, step: {step}
")
        notify_on_update(description, step)
    except Exception as e:
        logging.error(f"Error logging update: {e}")
        notify_on_error(f"Error logging update: {e}")

# Function to send notifications
def send_notification(subject, message):
    try:
        email = os.getenv('EMAIL')
        password = os.getenv('PASSWORD')
        recipient = os.getenv('RECIPIENT')

        msg = MIMEMultipart()
        msg['From'] = email
        msg['To'] = recipient
        msg['Subject'] = subject

        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP('smtp.example.com', 587)
        server.starttls()
        server.login(email, password)
        server.sendmail(email, recipient, msg.as_string())
        server.quit()
    except Exception as e:
        logging.error(f"Error sending notification: {e}")

# Notify on updates and errors
def notify_on_update(description, step):
    send_notification("Model Update", f"Model updated with description: {description}, step: {step}")

def notify_on_error(error_message):
    send_notification("Error Notification", error_message)

# Function to monitor a directory for new files
def monitor_directory(directory, known_files):
    while True:
        try:
            current_files = set(os.listdir(directory))
            new_files = current_files - known_files
            if new_files:
                logging.info(f"New data detected: {new_files}")
                process_new_data(new_files)
                known_files = current_files
            time.sleep(10)  # Check every 10 seconds
        except Exception as e:
            logging.error(f"Error in monitoring directory: {e}")
            notify_on_error(f"Error in monitoring directory: {e}")

# Initialize known files set and start monitoring
known_files = set(os.listdir(directory_to_watch))
monitor_directory(directory_to_watch, known_files)
