# NSE EOD Data Scraper 📊

Automated system for downloading End-of-Day (EOD) data from National Stock Exchange of India (NSE).

## 📦 Installation

### Prerequisites
- Python 3.11+
- Git
- Virtual environment support

### Setup Steps


# 1. Clone repository
```
git clone https://github.com/yourusername/zynapse-capital.git
cd zynapse-capital
```

# 2. Create virtual environment
```python -m venv venv```

# 3. Activate virtual environment
```
source venv/bin/activate  # Linux/macOS/WSL
# or
venv\Scripts\activate  # Windows
```

# 4. Install dependencies
```pip install -r requirements.txt```

### Dependencies

Create `requirements.txt`:
```
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
schedule>=1.2.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
python-dateutil>=2.8.2
pytz>=2023.3
```

### Verify Installation

```
# Test downloader
python scraper/downloader.py --date 2025-12-23

# Check calendar
python scraper/nse_calendar.py --list 90
```

## 🚀 Quick Start

```
# Download last 7 days
python scraper/downloader.py --days 7

# Start daily automation (runs at 6 PM)
python scraper/scheduler.py

# Run once immediately
python scraper/scheduler.py --run-now
```

## 📁 Project Structure

```
zynapse-capital/
├── scraper/
│   ├── downloader.py       # Core download engine
│   ├── scheduler.py        # Daily automation
│   └── nse_calendar.py     # Holiday calendar
├── data/
│   ├── raw/               # Downloaded files (by date)
│   └── cache/             # Holiday cache
├── logs/                  # Application logs
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

### Data downloaded daily:
- Equity Bhavcopy (OHLCV)
- Delivery Data
- F&O Bhavcopy
- Indices
- Participant OI & Volume
- Security Master

### Default settings:
- **Time:** 6:00 PM IST
- **Storage:** `data/raw/YYYY-MM-DD/`
- **Logs:** `logs/`
- **Auto-skip:** Weekends & holidays

## 🐧 Production Setup (Optional)

### Linux/WSL - Systemd Service

```
# Create service file
sudo nano /etc/systemd/system/nse-downloader.service
```

```
[Unit]
Description=NSE EOD Data Downloader
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/zynapse-capital
Environment="PATH=/path/to/zynapse-capital/venv/bin"
ExecStart=/path/to/zynapse-capital/venv/bin/python scraper/scheduler.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable nse-downloader
sudo systemctl start nse-downloader
sudo systemctl status nse-downloader
```

## 📞 Support

- **Issues:** GitHub Issues
- **Docs:** See code comments

---

**Made for Indian stock market data collection**
```

***
