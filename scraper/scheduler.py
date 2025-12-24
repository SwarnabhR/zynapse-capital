"""
Daily NSE Data Scheduler
Runs automatically at 6 PM IST (after market close + 30min buffer)
"""
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from downloader import NSEDownloader


class NSEScheduler:
    """Automated daily scheduler for NSE data downloads"""
    
    def __init__(self, download_time="18:00", log_dir='logs'):
        """
        Initialize scheduler
        
        Args:
            download_time: Time to run downloads (24-hour format, e.g., "18:00")
            log_dir: Directory for scheduler logs
        """
        self.download_time = download_time
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize downloader
        self.downloader = NSEDownloader()
        
        self.logger.info(f"Scheduler initialized - Daily run at {download_time} IST")
    
    def setup_logging(self):
        """Configure scheduler logging"""
        log_file = self.log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"
        
        self.logger = logging.getLogger('NSEScheduler')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
    
    def download_today(self):
        """Download today's EOD data"""
        today = datetime.now()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🕒 SCHEDULED RUN: {today.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*60}\n")
        
        # Skip weekends
        if today.weekday() >= 5:
            self.logger.info("⏭️  Skipping: Weekend (no trading)")
            return
        
        # Download today's data
        try:
            results = self.downloader.download_date(today, skip_existing=True)
            
            if all(results.values()):
                self.logger.info("✅ Scheduled download completed successfully")
            else:
                failed = [k for k, v in results.items() if not v]
                self.logger.warning(f"⚠️  Partial failure - Failed: {', '.join(failed)}")
        
        except Exception as e:
            self.logger.error(f"❌ Scheduled download failed: {e}")
    
    def download_yesterday(self):
        """Download yesterday's data (fallback/recovery)"""
        yesterday = datetime.now() - timedelta(days=1)
        
        # Skip if weekend
        if yesterday.weekday() >= 5:
            self.logger.info("⏭️  Skipping yesterday: Weekend")
            return
        
        self.logger.info(f"📅 Downloading yesterday's data: {yesterday.strftime('%Y-%m-%d')}")
        
        try:
            results = self.downloader.download_date(yesterday, skip_existing=True)
            
            if all(results.values()):
                self.logger.info("✅ Yesterday's data downloaded successfully")
            else:
                failed = [k for k, v in results.items() if not v]
                self.logger.warning(f"⚠️  Partial failure - Failed: {', '.join(failed)}")
        
        except Exception as e:
            self.logger.error(f"❌ Yesterday's download failed: {e}")
    
    def backfill_missing(self, days=30):
        """Check and download any missing dates from last N days"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔍 BACKFILL CHECK: Last {days} days")
        self.logger.info(f"{'='*60}\n")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            results = self.downloader.download_date_range(
                start_date, 
                end_date, 
                skip_existing=True,
                skip_weekends=True
            )
            
            self.logger.info("✅ Backfill check completed")
            
        except Exception as e:
            self.logger.error(f"❌ Backfill failed: {e}")
    
    def run_once(self):
        """Run download immediately (for testing)"""
        self.logger.info("🚀 Manual trigger - Running now")
        self.download_today()
    
    def start(self, backfill_on_start=True):
        """
        Start the scheduler
        
        Args:
            backfill_on_start: Run backfill check on startup
        """
        self.logger.info(f"\n{'#'*60}")
        self.logger.info("🤖 NSE DATA SCHEDULER STARTED")
        self.logger.info(f"{'#'*60}\n")
        self.logger.info(f"⏰ Daily download time: {self.download_time} IST")
        self.logger.info(f"📁 Data directory: {self.downloader.base_data_dir}")
        self.logger.info(f"📝 Log directory: {self.log_dir}")
        
        # Backfill missing data on startup
        if backfill_on_start:
            self.logger.info("\n🔄 Running initial backfill check...")
            self.backfill_missing(days=7)
        
        # Schedule daily download
        schedule.every().day.at(self.download_time).do(self.download_today)
        
        self.logger.info(f"\n✅ Scheduler is running - Next run: {self.download_time}")
        self.logger.info("Press Ctrl+C to stop\n")
        
        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Scheduler stopped by user")


def main():
    """CLI for scheduler"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NSE Data Scheduler')
    parser.add_argument('--time', type=str, default='18:00', 
                        help='Daily download time (HH:MM format)')
    parser.add_argument('--run-now', action='store_true', 
                        help='Run download immediately and exit')
    parser.add_argument('--backfill', type=int, 
                        help='Backfill last N days and exit')
    parser.add_argument('--no-startup-backfill', action='store_true',
                        help='Skip backfill check on startup')
    
    args = parser.parse_args()
    
    scheduler = NSEScheduler(download_time=args.time)
    
    try:
        if args.run_now:
            # Run once and exit
            scheduler.run_once()
        
        elif args.backfill:
            # Backfill and exit
            scheduler.backfill_missing(days=args.backfill)
        
        else:
            # Start continuous scheduler
            scheduler.start(backfill_on_start=not args.no_startup_backfill)
    
    except Exception as e:
        scheduler.logger.error(f"Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()