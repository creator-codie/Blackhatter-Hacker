
# Continuous Monitoring and Auto-Update

This project monitors a directory for new data, updates a machine learning model incrementally, and logs the updates.

## Features

- Continuous monitoring of a specified directory
- Incremental model updates
- Logging and error handling
- Email notifications for updates and errors

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the email settings in `monitor_and_update.py`.

## Usage

Run the monitoring script:
```bash
python monitor_and_update.py
```

## Configuration

For Unix-based systems, you can set up the script as a systemd service.
