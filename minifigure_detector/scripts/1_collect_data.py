import os
import time
import logging
import requests
import yaml
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

class MinifigureDataCollector:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup paths and API
        self.api_key = os.getenv('REBRICKABLE_API_KEY')
        self.base_url = 'https://rebrickable.com/api/v3/lego'
        self.base_dir = Path(os.getenv('DATASET_PATH', 'dataset'))
        
        if not self.api_key:
            raise ValueError("API key not found. Please set REBRICKABLE_API_KEY in .env file")

    def get_official_minifigs(self, page_size=50, min_year=1975, limit=None, skip_existing=True):
        """
        Get official LEGO minifigures with resume capability
        
        Args:
            page_size (int): Number of results per page
            min_year (int): Minimum year to filter results
            limit (int): Maximum number of minifigures to collect
            skip_existing (bool): Skip minifigures that are already in the dataset
            
        Returns:
            list: Collection of minifigure data
        """
        headers = {'Authorization': f'key {self.api_key}'}
        minifigs = []
        page = 1
        
        # Get list of already processed minifigs
        existing_minifigs = set()
        if skip_existing:
            metadata_dir = self.base_dir / 'metadata'
            if metadata_dir.exists():
                existing_minifigs = {p.stem for p in metadata_dir.glob('*.yaml')}
            self.logger.info(f"Found {len(existing_minifigs)} existing minifigures")
        
        self.logger.info(f"Collecting official minifigures from year {min_year} onwards...")
        pbar = tqdm(desc="Collecting minifigures", unit="minifigs")
        
        try:
            while True:
                if limit and len(minifigs) >= limit:
                    minifigs = minifigs[:limit]
                    break
                
                params = {
                    'page_size': min(page_size, limit - len(minifigs) if limit else page_size),
                    'page': page,
                    'min_year': min_year,
                    'ordering': 'year'
                }
                
                # Get main minifigs list
                try:
                    response = requests.get(
                        f"{self.base_url}/minifigs/",
                        headers=headers,
                        params=params,
                        timeout=10
                    )
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"API request failed: {e}")
                    break
                
                data = response.json()
                if not data.get('results'):
                    break
                    
                results = data['results']
                
                # Filter for official minifigures and skip existing ones
                for fig in results:
                    if not fig['set_num'].startswith('fig-'):
                        continue
                        
                    # Skip if already processed
                    if skip_existing and fig['set_num'] in existing_minifigs:
                        continue

                    try:
                        # First try getting year from the minifig's own details
                        minifig_response = requests.get(
                            f"{self.base_url}/minifigs/{fig['set_num']}/",
                            headers=headers,
                            timeout=10
                        )
                        minifig_response.raise_for_status()
                        minifig_data = minifig_response.json()
                        
                        # Try to get year from the minifig's details
                        if minifig_data.get('year'):
                            try:
                                fig['year'] = int(minifig_data['year'])
                                self.logger.debug(f"Found year {fig['year']} for {fig['set_num']} from minifig details")
                            except (ValueError, TypeError):
                                fig['year'] = None

                        # If no year found, try getting it from the sets
                        if not fig.get('year'):
                            # Get all sets that contain this minifig
                            sets_response = requests.get(
                                f"{self.base_url}/minifigs/{fig['set_num']}/sets/",
                                headers=headers,
                                timeout=10
                            )
                            sets_response.raise_for_status()
                            sets_data = sets_response.json()

                            years = []
                            if sets_data.get('results'):
                                for set_info in sets_data['results']:
                                    # Make another request to get detailed set info
                                    set_detail_response = requests.get(
                                        f"{self.base_url}/sets/{set_info['set_num']}/",
                                        headers=headers,
                                        timeout=10
                                    )
                                    if set_detail_response.status_code == 200:
                                        set_detail = set_detail_response.json()
                                        if set_detail.get('year'):
                                            try:
                                                years.append(int(set_detail['year']))
                                            except (ValueError, TypeError):
                                                continue

                            if years:
                                fig['year'] = min(years)  # First appearance year
                                self.logger.debug(f"Found year {fig['year']} for {fig['set_num']} from sets")
                            else:
                                fig['year'] = 'unknown'

                        # Log success or failure of year detection
                        if fig.get('year') and fig['year'] != 'unknown':
                            self.logger.info(f"✓ {fig['set_num']} - {fig['name']}: Year {fig['year']} found")
                        else:
                            self.logger.warning(f"⚠ {fig['set_num']} - {fig['name']}: No year found")

                        # Add delay to respect API rate limits
                        time.sleep(1)

                    except requests.exceptions.RequestException as e:
                        self.logger.warning(f"Failed to get details for {fig['set_num']}: {e}")
                        fig['year'] = 'unknown'
                        
                    minifigs.append(fig)
                    
                    if limit and len(minifigs) >= limit:
                        break
                
                pbar.update(len(minifigs) - pbar.n)
                
                if limit and len(minifigs) >= limit:
                    minifigs = minifigs[:limit]
                    break
                
                if not data.get('next'):
                    break
                
                page += 1
                time.sleep(1)  # Delay between pages
        
        except Exception as e:
            self.logger.error(f"Unexpected error during minifigure collection: {e}")
        finally:
            pbar.close()
        
        if not minifigs and not existing_minifigs:
            raise ValueError("No minifigures found! Please check your API key and internet connection.")
            
        self.logger.info(f"Found {len(minifigs)} new minifigures")
        return minifigs
    def download_image(self, url, path):
        """Download image from URL"""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                path.write_bytes(response.content)
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Failed to download image {url}: {e}")
            return False

    def save_metadata(self, minifig):
        """Save minifigure metadata"""
        metadata_file = self.base_dir / 'metadata' / f"{minifig['set_num']}.yaml"
        metadata = {
            'set_num': minifig.get('set_num', ''),
            'name': minifig.get('name', ''),
            'year': minifig.get('year', 'unknown'),
            'num_parts': minifig.get('num_parts', 0),
            'url': minifig.get('set_url', '')
        }
        
        # Get additional details if available
        try:
            headers = {'Authorization': f'key {self.api_key}'}
            response = requests.get(
                f"{self.base_url}/minifigs/{minifig['set_num']}/",
                headers=headers
            )
            if response.status_code == 200:
                details = response.json()
                metadata.update({
                    'theme': details.get('theme_id', ''),
                    'category': details.get('category_id', ''),
                    'designer': details.get('designer_id', '')
                })
        except Exception as e:
            self.logger.warning(f"Could not fetch additional details for {minifig['set_num']}: {e}")
        
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f)

    def collect_data(self, limit=None):
        """Collect minifigure data with resume capability"""
        try:
            # Create directories if they don't exist
            for dir_name in ['images', 'metadata', 'labels']:
                dir_path = self.base_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Load existing summary if it exists
            summary_file = self.base_dir / 'dataset_summary.yaml'
            existing_summary = {}
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    existing_summary = yaml.safe_load(f)
            
            # Get new minifigures
            minifigs = self.get_official_minifigs(limit=limit, skip_existing=True)
            
            if not minifigs and not existing_summary:
                self.logger.info("No new minifigures to process")
                return
            
            # Update summary
            total_minifigs = len(minifigs) + existing_summary.get('total_minifigs', 0)
            
            # Get year range including existing data
            start_year = existing_summary.get('year_range', '').split('-')[0] if existing_summary else "unknown"
            end_year = existing_summary.get('year_range', '').split('-')[1] if existing_summary else "unknown"
            
            for fig in minifigs:
                if fig.get('year'):
                    if start_year == "unknown" or fig['year'] < int(start_year):
                        start_year = fig['year']
                    if end_year == "unknown" or fig['year'] > int(end_year):
                        end_year = fig['year']
            
            # Create/update summary
            summary = {
                'total_minifigs': total_minifigs,
                'year_range': f"{start_year}-{end_year}",
                'collection_date': time.strftime('%Y-%m-%d'),
                'downloaded_images': existing_summary.get('downloaded_images', 0)
            }
            
            # Download new images and save metadata
            successful_downloads = 0
            with tqdm(total=len(minifigs), desc="Downloading images") as pbar:
                for minifig in minifigs:
                    if minifig.get('set_img_url'):
                        image_path = self.base_dir / 'images' / f"{minifig['set_num']}.jpg"
                        
                        if not image_path.exists():
                            if self.download_image(minifig['set_img_url'], image_path):
                                self.save_metadata(minifig)
                                successful_downloads += 1
                            time.sleep(0.5)
                    pbar.update(1)
            
            # Update summary with new downloads
            summary['downloaded_images'] += successful_downloads
            with open(summary_file, 'w') as f:
                yaml.dump(summary, f)
            
            self.logger.info(f"""
Dataset collection completed:
- Total minifigures: {summary['total_minifigs']}
- New minifigures added: {len(minifigs)}
- Total images downloaded: {summary['downloaded_images']}
- Year range: {summary['year_range']}
            """)
            
        except Exception as e:
            self.logger.error(f"Error during data collection: {e}")
            raise

def main():
    # Get number of minifigures to collect from user input
    limit_input = input("Enter number of minifigures to collect (press Enter for all): ").strip()
    limit = int(limit_input) if limit_input else None
    
    # Initialize and run collector
    collector = MinifigureDataCollector()
    collector.collect_data(limit=limit)

if __name__ == "__main__":
    main()