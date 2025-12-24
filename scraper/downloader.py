"""
NSE Complete EOD Data Downloader - Production Version
Features: Date ranges, retry logic, skip existing, logging, progress tracking
"""
import requests
from datetime import datetime, timedelta
from pathlib import Path
import zipfile
import gzip
import shutil
import time
import logging
from typing import Dict, List, Tuple


class NSEDownloader:
    """Production-ready NSE data downloader with automation features"""
    
    BASE_URL = "https://nsearchives.nseindia.com"
    
    URLS = {
        'bhavcopy': '/archives/equities/bhavcopy/pr/PR{date_ddmmyy}.zip',
        'delivery': '/archives/equities/mto/MTO_{date_ddmmyyyy}.DAT',
        'security_master': '/content/cm/interop/NSE_CM_security_{date_ddmmyyyy}.csv.gz',
        'fo_bhavcopy': '/products/content/sec_bhavdata_full_{date_ddmmyyyy}.csv',
        'indices': '/content/indices/ind_close_all_{date_ddmmyyyy}.csv',
        'participant_oi': '/content/nsccl/fao_participant_oi_{date_ddmmyyyy}.csv',
        'participant_vol': '/content/nsccl/fao_participant_vol_{date_ddmmyyyy}.csv',
    }
    
    def __init__(self, data_dir='data/raw', log_dir='logs', max_retries=3):
        """
        Initialize downloader with automation features
        
        Args:
            data_dir: Base directory for data storage
            log_dir: Directory for log files
            max_retries: Maximum retry attempts for failed downloads
        """
        self.base_data_dir = Path(data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = 2  # seconds
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.logger.info("NSE Downloader initialized")
    
    def setup_logging(self):
        """Configure logging to file and console"""
        log_file = self.log_dir / f"downloads_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Create logger
        self.logger = logging.getLogger('NSEDownloader')
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
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
    
    def setup_date_directories(self, date):
        """Create date-specific directory structure"""
        date_folder = self.base_data_dir / date.strftime('%Y-%m-%d')
        date_folder.mkdir(parents=True, exist_ok=True)
        
        self.bhavcopy_dir = date_folder / 'bhavcopy'
        self.delivery_dir = date_folder / 'delivery'
        self.security_master_dir = date_folder / 'security_master'
        self.derivatives_dir = date_folder / 'derivatives'
        self.indices_dir = date_folder / 'indices'
        self.participant_oi_dir = date_folder / 'participant_oi'
        
        for directory in [self.bhavcopy_dir, self.delivery_dir, self.security_master_dir,
                          self.derivatives_dir, self.indices_dir, self.participant_oi_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def format_date(self, date, format_type='ddmmyyyy'):
        """Convert datetime to NSE filename format"""
        if format_type == 'ddmmyy':
            return date.strftime('%d%m%y')
        elif format_type == 'ddmmyyyy':
            return date.strftime('%d%m%Y')
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def file_exists(self, filepath):
        """Check if file already exists and is non-empty"""
        return filepath.exists() and filepath.stat().st_size > 0
    
    def extract_zip(self, zip_path, extract_to):
        """Extract ZIP file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            zip_path.unlink()
            return True
        except Exception as e:
            self.logger.error(f"ZIP extraction failed: {e}")
            return False
    
    def decompress_gz(self, gz_path, output_path):
        """Decompress GZ file"""
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()
            return True
        except Exception as e:
            self.logger.error(f"GZ decompression failed: {e}")
            return False
    
    def download_file_with_retry(self, url, filepath, skip_existing=True):
        """
        Download file with retry logic
        
        Args:
            url: URL to download from
            filepath: Local path to save file
            skip_existing: Skip if file already exists
        
        Returns:
            bool: Success status
        """
        # Check if file exists
        if skip_existing and self.file_exists(filepath):
            self.logger.info(f"⏭️  Skipped (exists): {filepath.name}")
            return True
        
        # Retry loop
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(f"⬇️  Downloading: {filepath.name} (attempt {attempt}/{self.max_retries})")
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                file_size = len(response.content) / 1024
                self.logger.info(f"✅ Downloaded: {filepath.name} ({file_size:.2f} KB)")
                return True
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"❌ Attempt {attempt} failed: {e}")
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)  # Exponential backoff
                else:
                    self.logger.error(f"💥 All retries failed for: {filepath.name}")
                    return False
        
        return False
    
    def download_bhavcopy(self, date, skip_existing=True):
        """Download equity bhavcopy"""
        date_str = self.format_date(date, 'ddmmyy')
        url = self.BASE_URL + self.URLS['bhavcopy'].format(date_ddmmyy=date_str)
        zip_path = self.bhavcopy_dir / f"PR{date_str}.zip"
        
        # Check if already extracted
        if skip_existing and list(self.bhavcopy_dir.glob('*.csv')):
            self.logger.info("⏭️  Skipped: bhavcopy (files exist)")
            return True
        
        if not self.download_file_with_retry(url, zip_path, skip_existing=False):
            return False
        
        return self.extract_zip(zip_path, self.bhavcopy_dir)
    
    def download_delivery(self, date, skip_existing=True):
        """Download delivery data"""
        date_str = self.format_date(date, 'ddmmyyyy')
        url = self.BASE_URL + self.URLS['delivery'].format(date_ddmmyyyy=date_str)
        dat_path = self.delivery_dir / f"MTO_{date_str}.DAT"
        
        return self.download_file_with_retry(url, dat_path, skip_existing)
    
    def download_security_master(self, date, skip_existing=True):
        """Download security master"""
        date_str = self.format_date(date, 'ddmmyyyy')
        url = self.BASE_URL + self.URLS['security_master'].format(date_ddmmyyyy=date_str)
        gz_path = self.security_master_dir / f"NSE_CM_security_{date_str}.csv.gz"
        csv_path = self.security_master_dir / f"NSE_CM_security_{date_str}.csv"
        
        if skip_existing and self.file_exists(csv_path):
            self.logger.info(f"⏭️  Skipped (exists): {csv_path.name}")
            return True
        
        if not self.download_file_with_retry(url, gz_path, skip_existing=False):
            return False
        
        return self.decompress_gz(gz_path, csv_path)
    
    def download_fo_bhavcopy(self, date, skip_existing=True):
        """Download F&O data"""
        date_str = self.format_date(date, 'ddmmyyyy')
        url = self.BASE_URL + self.URLS['fo_bhavcopy'].format(date_ddmmyyyy=date_str)
        csv_path = self.derivatives_dir / f"fo_bhav_{date_str}.csv"
        
        return self.download_file_with_retry(url, csv_path, skip_existing)
    
    def download_indices(self, date, skip_existing=True):
        """Download indices data"""
        date_str = self.format_date(date, 'ddmmyyyy')
        url = self.BASE_URL + self.URLS['indices'].format(date_ddmmyyyy=date_str)
        csv_path = self.indices_dir / f"indices_{date_str}.csv"
        
        return self.download_file_with_retry(url, csv_path, skip_existing)
    
    def download_participant_oi(self, date, skip_existing=True):
        """Download participant OI"""
        date_str = self.format_date(date, 'ddmmyyyy')
        url = self.BASE_URL + self.URLS['participant_oi'].format(date_ddmmyyyy=date_str)
        csv_path = self.participant_oi_dir / f"participant_oi_{date_str}.csv"
        
        return self.download_file_with_retry(url, csv_path, skip_existing)
    
    def download_participant_vol(self, date, skip_existing=True):
        """Download participant volume"""
        date_str = self.format_date(date, 'ddmmyyyy')
        url = self.BASE_URL + self.URLS['participant_vol'].format(date_ddmmyyyy=date_str)
        csv_path = self.participant_oi_dir / f"participant_vol_{date_str}.csv"
        
        return self.download_file_with_retry(url, csv_path, skip_existing)
    
    def download_date(self, date, skip_existing=True):
        """
        Download all files for a single date
        
        Args:
            date: datetime object or string (YYYY-MM-DD)
            skip_existing: Skip files that already exist
        
        Returns:
            dict: Results for each file type
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📅 Processing: {date.strftime('%d-%b-%Y (%A)')}")
        self.logger.info(f"{'='*60}")
        
        # Setup directories
        self.setup_date_directories(date)
        
        # Download all files
        results = {
            'bhavcopy': self.download_bhavcopy(date, skip_existing),
            'delivery': self.download_delivery(date, skip_existing),
            'security_master': self.download_security_master(date, skip_existing),
            'fo_bhavcopy': self.download_fo_bhavcopy(date, skip_existing),
            'indices': self.download_indices(date, skip_existing),
            'participant_oi': self.download_participant_oi(date, skip_existing),
            'participant_vol': self.download_participant_vol(date, skip_existing),
        }
        
        success_count = sum(results.values())
        total_count = len(results)
        
        self.logger.info(f"✅ {success_count}/{total_count} files successful")
        
        return results
    
    def download_date_range(self, start_date, end_date, skip_existing=True, skip_weekends=True):
        """
        Download data for a range of dates
        
        Args:
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
            skip_existing: Skip files that already exist
            skip_weekends: Skip Saturday and Sunday
        
        Returns:
            dict: Summary of downloads
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        self.logger.info(f"\n{'#'*60}")
        self.logger.info(f"🚀 BULK DOWNLOAD: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"{'#'*60}\n")
        
        current_date = start_date
        all_results = {}
        
        while current_date <= end_date:
            # Skip weekends if requested
            if skip_weekends and current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                self.logger.info(f"⏭️  Skipping weekend: {current_date.strftime('%Y-%m-%d (%A)')}")
                current_date += timedelta(days=1)
                continue
            
            # Download for this date
            date_str = current_date.strftime('%Y-%m-%d')
            all_results[date_str] = self.download_date(current_date, skip_existing)
            
            current_date += timedelta(days=1)
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, all_results):
        """Print download summary"""
        self.logger.info(f"\n{'#'*60}")
        self.logger.info("📊 DOWNLOAD SUMMARY")
        self.logger.info(f"{'#'*60}\n")
        
        total_dates = len(all_results)
        successful_dates = sum(1 for results in all_results.values() if all(results.values()))
        
        self.logger.info(f"📅 Total dates processed: {total_dates}")
        self.logger.info(f"✅ Fully successful dates: {successful_dates}")
        self.logger.info(f"⚠️  Partially failed dates: {total_dates - successful_dates}")
        
        # Show failed dates
        failed_dates = [date for date, results in all_results.items() if not all(results.values())]
        if failed_dates:
            self.logger.warning(f"\n❌ Dates with failures:")
            for date in failed_dates:
                failed_files = [k for k, v in all_results[date].items() if not v]
                self.logger.warning(f"   {date}: {', '.join(failed_files)}")


def main():
    """Command-line interface with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NSE EOD Data Downloader')
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, help='Start date for range (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date for range (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, help='Download last N days')
    parser.add_argument('--force', action='store_true', help='Re-download existing files')
    parser.add_argument('--include-weekends', action='store_true', help='Include weekends')
    
    args = parser.parse_args()
    
    downloader = NSEDownloader()
    
    try:
        if args.date:
            # Single date
            downloader.download_date(args.date, skip_existing=not args.force)
        
        elif args.start and args.end:
            # Date range
            downloader.download_date_range(args.start, args.end, 
                                           skip_existing=not args.force,
                                           skip_weekends=not args.include_weekends)
        
        elif args.days:
            # Last N days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            downloader.download_date_range(start_date, end_date,
                                           skip_existing=not args.force,
                                           skip_weekends=not args.include_weekends)
        
        else:
            # Default: yesterday's data
            yesterday = datetime.now() - timedelta(days=1)
            downloader.download_date(yesterday, skip_existing=not args.force)
    
    except Exception as e:
        downloader.logger.error(f"Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()