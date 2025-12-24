"""
NSE Trading Holiday Calendar
Maintains list of market holidays and checks trading days
"""
import requests
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import List, Set


class NSECalendar:
    """NSE Trading Calendar - Market holidays and trading days"""
    
    # NSE Holiday List API (official)
    HOLIDAY_API = "https://www.nseindia.com/api/holiday-master?type=trading"
    
    # Manual holiday list as fallback (2025-2026)
    HOLIDAYS_2025_2026 = {
        # 2025
        '2025-01-26': 'Republic Day',
        '2025-03-14': 'Holi',
        '2025-03-31': 'Id-Ul-Fitr',
        '2025-04-10': 'Mahavir Jayanti',
        '2025-04-14': 'Dr. Baba Saheb Ambedkar Jayanti',
        '2025-04-18': 'Good Friday',
        '2025-05-01': 'Maharashtra Day',
        '2025-06-07': 'Id-Ul-Adha (Bakri Id)',
        '2025-07-06': 'Moharram',
        '2025-08-15': 'Independence Day',
        '2025-08-27': 'Ganesh Chaturthi',
        '2025-09-05': 'Eid-e-Milad',
        '2025-10-02': 'Mahatma Gandhi Jayanti',
        '2025-10-21': 'Dussehra',
        '2025-11-01': 'Diwali-Laxmi Pujan',
        '2025-11-04': 'Diwali-Balipratipada',
        '2025-11-05': 'Gurunanak Jayanti',
        '2025-12-25': 'Christmas',
        
        # 2026
        '2026-01-26': 'Republic Day',
        '2026-03-03': 'Holi',
        '2026-03-21': 'Id-Ul-Fitr',
        '2026-03-30': 'Mahavir Jayanti',
        '2026-04-03': 'Good Friday',
        '2026-04-06': 'Ram Navami',
        '2026-04-14': 'Dr. Baba Saheb Ambedkar Jayanti',
        '2026-05-01': 'Maharashtra Day',
        '2026-05-27': 'Id-Ul-Adha (Bakri Id)',
        '2026-06-16': 'Moharram',
        '2026-08-15': 'Independence Day',
        '2026-08-26': 'Eid-e-Milad',
        '2026-09-16': 'Ganesh Chaturthi',
        '2026-10-02': 'Mahatma Gandhi Jayanti',
        '2026-10-10': 'Dussehra',
        '2026-10-20': 'Diwali-Laxmi Pujan',
        '2026-10-22': 'Diwali-Balipratipada',
        '2026-11-25': 'Gurunanak Jayanti',
        '2026-12-25': 'Christmas',
    }
    
    def __init__(self, cache_dir='data/cache'):
        """
        Initialize NSE Calendar
        
        Args:
            cache_dir: Directory to cache holiday data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / 'nse_holidays.json'
        self.holidays = set()
        
        # Setup logging
        self.logger = logging.getLogger('NSECalendar')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Load holidays
        self.load_holidays()
    
    def load_holidays(self):
        """Load holidays from cache or fetch from API"""
        # Try loading from cache first
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.holidays = set(data['holidays'])
                    cache_date = data.get('updated', 'Unknown')
                    self.logger.info(f"✅ Loaded {len(self.holidays)} holidays from cache (updated: {cache_date})")
                    return
            except Exception as e:
                self.logger.warning(f"Cache load failed: {e}")
        
        # Fetch from API
        self.fetch_holidays_from_api()
        
        # Fallback to manual list if API fails
        if not self.holidays:
            self.logger.warning("API fetch failed - Using manual holiday list")
            self.holidays = set(self.HOLIDAYS_2025_2026.keys())
            self.save_cache()
    
    def fetch_holidays_from_api(self):
        """Fetch holidays from NSE API"""
        try:
            self.logger.info("🔄 Fetching holidays from NSE API...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            session = requests.Session()
            
            # First request to get cookies
            session.get('https://www.nseindia.com', headers=headers, timeout=10)
            
            # Get holidays
            response = session.get(self.HOLIDAY_API, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse holidays from API response
            holiday_dates = set()
            
            if 'CM' in data:  # Cash Market holidays
                for holiday in data['CM']:
                    date_str = holiday.get('tradingDate', '')
                    if date_str:
                        # Convert DD-MMM-YYYY to YYYY-MM-DD
                        try:
                            date_obj = datetime.strptime(date_str, '%d-%b-%Y')
                            holiday_dates.add(date_obj.strftime('%Y-%m-%d'))
                        except ValueError:
                            continue
            
            if holiday_dates:
                self.holidays = holiday_dates
                self.logger.info(f"✅ Fetched {len(self.holidays)} holidays from NSE API")
                self.save_cache()
            else:
                self.logger.warning("⚠️  No holidays found in API response")
        
        except Exception as e:
            self.logger.error(f"❌ API fetch failed: {e}")
    
    def save_cache(self):
        """Save holidays to cache"""
        try:
            data = {
                'holidays': list(self.holidays),
                'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"💾 Saved {len(self.holidays)} holidays to cache")
        
        except Exception as e:
            self.logger.error(f"Cache save failed: {e}")
    
    def is_holiday(self, date) -> bool:
        """
        Check if date is a market holiday
        
        Args:
            date: datetime object or string (YYYY-MM-DD)
        
        Returns:
            bool: True if holiday, False otherwise
        """
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date
        
        return date_str in self.holidays
    
    def is_weekend(self, date) -> bool:
        """
        Check if date is a weekend (Saturday/Sunday)
        
        Args:
            date: datetime object or string (YYYY-MM-DD)
        
        Returns:
            bool: True if weekend, False otherwise
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        return date.weekday() >= 5  # 5=Saturday, 6=Sunday
    
    def is_trading_day(self, date) -> bool:
        """
        Check if date is a trading day (not weekend, not holiday)
        
        Args:
            date: datetime object or string (YYYY-MM-DD)
        
        Returns:
            bool: True if trading day, False otherwise
        """
        return not (self.is_weekend(date) or self.is_holiday(date))
    
    def get_holiday_name(self, date) -> str:
        """
        Get holiday name for a given date
        
        Args:
            date: datetime object or string (YYYY-MM-DD)
        
        Returns:
            str: Holiday name or empty string
        """
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date
        
        return self.HOLIDAYS_2025_2026.get(date_str, '')
    
    def get_next_trading_day(self, date=None) -> datetime:
        """
        Get next trading day from given date
        
        Args:
            date: Starting date (default: today)
        
        Returns:
            datetime: Next trading day
        """
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        next_day = date + timedelta(days=1)
        
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        
        return next_day
    
    def get_previous_trading_day(self, date=None) -> datetime:
        """
        Get previous trading day from given date
        
        Args:
            date: Starting date (default: today)
        
        Returns:
            datetime: Previous trading day
        """
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        prev_day = date - timedelta(days=1)
        
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        
        return prev_day
    
    def get_trading_days_range(self, start_date, end_date) -> List[datetime]:
        """
        Get list of trading days in a date range
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            List of trading days
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def update_holidays(self):
        """Force update holidays from API"""
        self.logger.info("🔄 Force updating holidays from NSE...")
        self.fetch_holidays_from_api()
    
    def list_upcoming_holidays(self, days=90):
        """
        List upcoming holidays
        
        Args:
            days: Number of days to look ahead
        """
        today = datetime.now()
        end_date = today + timedelta(days=days)
        
        upcoming = []
        current = today
        
        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            if self.is_holiday(current):
                holiday_name = self.get_holiday_name(current)
                upcoming.append({
                    'date': date_str,
                    'day': current.strftime('%A'),
                    'holiday': holiday_name or 'Market Holiday'
                })
            current += timedelta(days=1)
        
        return upcoming


def main():
    """CLI for calendar management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NSE Trading Calendar')
    parser.add_argument('--check', type=str, help='Check if date is trading day (YYYY-MM-DD)')
    parser.add_argument('--update', action='store_true', help='Update holidays from NSE API')
    parser.add_argument('--list', type=int, default=90, help='List upcoming holidays (days ahead)')
    parser.add_argument('--next', action='store_true', help='Get next trading day')
    parser.add_argument('--prev', action='store_true', help='Get previous trading day')
    
    args = parser.parse_args()
    
    calendar = NSECalendar()
    
    if args.update:
        calendar.update_holidays()
    
    elif args.check:
        date = args.check
        is_trading = calendar.is_trading_day(date)
        is_weekend = calendar.is_weekend(date)
        is_holiday = calendar.is_holiday(date)
        
        print(f"\n📅 Date: {date}")
        print(f"Trading Day: {'✅ Yes' if is_trading else '❌ No'}")
        
        if is_weekend:
            print(f"Reason: Weekend")
        elif is_holiday:
            holiday_name = calendar.get_holiday_name(date)
            print(f"Reason: {holiday_name}")
    
    elif args.next:
        next_day = calendar.get_next_trading_day()
        print(f"\n📅 Next trading day: {next_day.strftime('%Y-%m-%d (%A)')}")
    
    elif args.prev:
        prev_day = calendar.get_previous_trading_day()
        print(f"\n📅 Previous trading day: {prev_day.strftime('%Y-%m-%d (%A)')}")
    
    else:
        # List upcoming holidays
        upcoming = calendar.list_upcoming_holidays(days=args.list)
        
        if upcoming:
            print(f"\n📅 Upcoming NSE Holidays ({len(upcoming)} days):\n")
            print(f"{'Date':<12} {'Day':<10} {'Holiday'}")
            print("=" * 60)
            for holiday in upcoming:
                print(f"{holiday['date']:<12} {holiday['day']:<10} {holiday['holiday']}")
        else:
            print(f"\n✅ No holidays in next {args.list} days")


if __name__ == '__main__':
    main()