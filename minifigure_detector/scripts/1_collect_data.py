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
        self.metadata_dir = self.base_dir / 'metadata'
        self.images_dir = self.base_dir / 'images' / 'train'
        
        # Create necessary directories
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.api_key:
            raise ValueError("API key not found. Please set REBRICKABLE_API_KEY in .env file")

    def get_existing_minifigs(self):
        """Get list of existing minifigure IDs from metadata directory"""
        if not self.metadata_dir.exists():
            return set()
        return {p.stem for p in self.metadata_dir.glob('*.yaml')}

    def get_official_minifigs(self, page_size=50, min_year=1975, limit=None):
        """Get official LEGO minifigures with pagination"""
        headers = {'Authorization': f'key {self.api_key}'}
        minifigs = []
        page = 1
        
        # Get existing minifigs to skip
        existing_minifigs = self.get_existing_minifigs()
        self.logger.info(f"Found {len(existing_minifigs)} existing minifigures")
        
        self.logger.info(f"Collecting official minifigures from year {min_year} onwards...")
        pbar = tqdm(desc="Collecting minifigures", unit="minifigs")
        
        while True:
            try:
                # Break if we've reached the limit
                if limit and len(minifigs) >= limit:
                    break
                
                # Make API request for minifigs list
                params = {
                    'page_size': page_size,
                    'page': page,
                    'min_year': min_year,
                    'ordering': 'year'
                }
                
                response = requests.get(
                    f"{self.base_url}/minifigs/",
                    headers=headers,
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get('results'):
                    break
                
                # Process results
                for fig in data['results']:
                    # Skip if we've reached the limit
                    if limit and len(minifigs) >= limit:
                        break
                    
                    # Skip if already exists
                    if fig['set_num'] in existing_minifigs:
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
                                fig['year'] = -1
                        else:
                            fig['year'] = -1
                        
                        # Add theme and category
                        fig['theme_id'] = minifig_data.get('theme_id', 'unknown')
                        fig['category_id'] = minifig_data.get('category_id', 'unknown')

                        # If no year found, try getting it from the sets
                        if fig['year'] == -1:
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

                        # Add delay to respect API rate limits
                        time.sleep(1)

                        # Log success or failure of year detection
                        if fig['year'] != -1:
                            self.logger.info(f"✓ {fig['set_num']} - {fig['name']}: Year {fig['year']} found")
                        else:
                            self.logger.warning(f"⚠ {fig['set_num']} - {fig['name']}: No year found")

                        minifigs.append(fig)
                        pbar.update(1)
                        
                    except requests.exceptions.RequestException as e:
                        self.logger.warning(f"Failed to get details for {fig['set_num']}: {e}")
                        continue
                    
                # Break if no more pages
                if not data.get('next'):
                    break
                
                page += 1
                time.sleep(1)  # Rate limiting between pages
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API request failed: {e}")
                break
            except Exception as e:
                self.logger.error(f"Error during minifigure collection: {e}")
                break
        
        pbar.close()
        return minifigs

    def download_image(self, url, path):
        """Download image from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            path.write_bytes(response.content)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to download image {url}: {e}")
            return False

    def save_metadata(self, minifig):
        """Save minifigure metadata"""
        try:
            metadata = {
                'minifigure': {
                    'set_num': minifig['set_num'],
                    'name': minifig['name'],
                    'year': minifig['year'],
                    'theme': minifig.get('theme_id', 'unknown'),
                    'category': 'official'
                }
            }
            
            metadata_file = self.metadata_dir / f"{minifig['set_num']}.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f)
            
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save metadata for {minifig['set_num']}: {e}")
            return False

    def update_incomplete_minifigs(self):
        """Update existing minifigures with missing or -1 years"""
        headers = {'Authorization': f'key {self.api_key}'}
        updated_count = 0
        
        # Get all metadata files
        metadata_files = list(self.metadata_dir.glob('*.yaml'))
        
        # Filter for minifigs with year == -1 or 'unknown'
        incomplete_minifigs = []
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = yaml.safe_load(f)
                minifig = metadata.get('minifigure', {})
                year = minifig.get('year')
                if year == -1 or year == 'unknown':
                    incomplete_minifigs.append((metadata_file, metadata))
        
        self.logger.info(f"Found {len(incomplete_minifigs)} minifigures with missing years (year = -1 or 'unknown')")
        
        with tqdm(incomplete_minifigs, desc="Updating minifigures") as pbar:
            for metadata_file, metadata in pbar:
                try:
                    set_num = metadata['minifigure']['set_num']
                    current_year = metadata['minifigure']['year']
                    
                    self.logger.info(f"Updating {set_num} (current year: {current_year})")
                    
                    # First try getting year from minifig details
                    try:
                        minifig_response = requests.get(
                            f"{self.base_url}/minifigs/{set_num}/",
                            headers=headers,
                            timeout=10
                        )
                        minifig_response.raise_for_status()
                        minifig_data = minifig_response.json()
                        
                        if minifig_data.get('year'):
                            try:
                                new_year = int(minifig_data['year'])
                                metadata['minifigure']['year'] = new_year
                                self.logger.info(f"Found year {new_year} for {set_num} from minifig details")
                                updated_count += 1
                                
                                # Save updated metadata
                                with open(metadata_file, 'w') as f:
                                    yaml.dump(metadata, f)
                                continue
                            except (ValueError, TypeError):
                                pass
                        
                        # If still no year, try getting from sets
                        time.sleep(1)  # Rate limiting
                        sets_response = requests.get(
                            f"{self.base_url}/minifigs/{set_num}/sets/",
                            headers=headers,
                            timeout=10
                        )
                        sets_response.raise_for_status()
                        sets_data = sets_response.json()
                        
                        years = []
                        if sets_data.get('results'):
                            for set_info in sets_data['results']:
                                time.sleep(1)  # Rate limiting
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
                            new_year = min(years)  # First appearance year
                            metadata['minifigure']['year'] = new_year
                            self.logger.info(f"Found year {new_year} for {set_num} from sets")
                            updated_count += 1
                        else:
                            # If still no year found, ensure it's set to -1
                            metadata['minifigure']['year'] = -1
                            self.logger.warning(f"No year found for {set_num}, setting to -1")
                        
                        # Save updated metadata
                        with open(metadata_file, 'w') as f:
                            yaml.dump(metadata, f)
                    
                    except requests.exceptions.RequestException as e:
                        self.logger.warning(f"Failed to update {set_num}: {e}")
                        # Ensure year is set to -1 if update fails
                        metadata['minifigure']['year'] = -1
                        with open(metadata_file, 'w') as f:
                            yaml.dump(metadata, f)
                        continue
                    
                    time.sleep(1)  # Rate limiting
                
                except Exception as e:
                    self.logger.warning(f"Error processing {metadata_file.name}: {e}")
                    continue
                
                pbar.update(1)
        
        # Print summary
        self.logger.info(f"""
Update Summary:
--------------
Total minifigures checked: {len(metadata_files)}
Minifigures with missing years: {len(incomplete_minifigs)}
Successfully updated: {updated_count}
Remaining with year = -1: {len(incomplete_minifigs) - updated_count}
""")
        
        return updated_count

    def validate_and_repair_dataset(self):
        """Validate and repair dataset, handling missing or incomplete metadata"""
        self.logger.info("Starting dataset validation and repair...")
        
        # Get all files
        image_files = list(self.images_dir.glob('*.png')) + list(self.images_dir.glob('*.jpg'))
        metadata_files = list(self.metadata_dir.glob('*.yaml'))
        
        # Create mappings
        image_ids = {img.stem for img in image_files}
        metadata_ids = {meta.stem for meta in metadata_files}
        
        # Find discrepancies
        images_without_metadata = image_ids - metadata_ids
        metadata_without_images = metadata_ids - image_ids
        
        self.logger.info(f"""
Found discrepancies:
- Images without metadata: {len(images_without_metadata)}
- Metadata without images: {len(metadata_without_images)}
""")

        # 1. Handle images without metadata
        if images_without_metadata:
            self.logger.info("Attempting to recreate metadata for images without metadata...")
            headers = {'Authorization': f'key {self.api_key}'}
            
            with tqdm(images_without_metadata, desc="Recreating metadata") as pbar:
                for img_id in pbar:
                    try:
                        # Try to get minifig details from API
                        minifig_response = requests.get(
                            f"{self.base_url}/minifigs/{img_id}/",
                            headers=headers,
                            timeout=10
                        )
                        
                        if minifig_response.status_code == 200:
                            minifig_data = minifig_response.json()
                            
                            # Get year using our existing year-finding logic
                            year = -1
                            if minifig_data.get('year'):
                                try:
                                    year = int(minifig_data['year'])
                                except (ValueError, TypeError):
                                    pass
                            
                            if year == -1:
                                # Try getting year from sets
                                sets_response = requests.get(
                                    f"{self.base_url}/minifigs/{img_id}/sets/",
                                    headers=headers,
                                    timeout=10
                                )
                                if sets_response.status_code == 200:
                                    sets_data = sets_response.json()
                                    years = []
                                    
                                    for set_info in sets_data.get('results', []):
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
                                        year = min(years)
                            
                            # Create metadata
                            metadata = {
                                'minifigure': {
                                    'set_num': img_id,
                                    'name': minifig_data.get('name', 'Unknown'),
                                    'year': year,
                                    'theme': minifig_data.get('theme_id', 'unknown'),
                                    'category': 'official'
                                }
                            }
                            
                            # Save metadata
                            metadata_path = self.metadata_dir / f"{img_id}.yaml"
                            with open(metadata_path, 'w') as f:
                                yaml.dump(metadata, f)
                                
                            self.logger.info(f"Created metadata for {img_id}")
                        else:
                            self.logger.warning(f"Could not get data for {img_id} from API")
                            
                        time.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {img_id}: {e}")
                        continue
                    
                    pbar.update(1)

        # 2. Option to remove orphaned metadata files
        if metadata_without_images:
            print("\nFound metadata files without corresponding images:")
            for meta_id in metadata_without_images:
                print(f"- {meta_id}")
            
            choice = input("\nWould you like to remove these orphaned metadata files? (y/n): ").strip().lower()
            if choice == 'y':
                for meta_id in metadata_without_images:
                    try:
                        (self.metadata_dir / f"{meta_id}.yaml").unlink()
                        self.logger.info(f"Removed orphaned metadata: {meta_id}")
                    except Exception as e:
                        self.logger.error(f"Error removing {meta_id}: {e}")

        # 3. Validate all remaining metadata files
        self.logger.info("Validating remaining metadata files...")
        invalid_metadata = []
        
        with tqdm(metadata_files, desc="Validating metadata") as pbar:
            for meta_file in pbar:
                try:
                    with open(meta_file, 'r') as f:
                        metadata = yaml.safe_load(f)
                    
                    # Check required fields
                    required_fields = ['set_num', 'name', 'year', 'theme', 'category']
                    if not all(field in metadata.get('minifigure', {}) for field in required_fields):
                        invalid_metadata.append(meta_file.stem)
                        
                except Exception as e:
                    self.logger.error(f"Error validating {meta_file.name}: {e}")
                    invalid_metadata.append(meta_file.stem)
                
                pbar.update(1)
        
        if invalid_metadata:
            self.logger.warning(f"Found {len(invalid_metadata)} invalid metadata files")
            print("\nInvalid metadata files:")
            for meta_id in invalid_metadata:
                print(f"- {meta_id}")
            
            choice = input("\nWould you like to attempt to repair these files? (y/n): ").strip().lower()
            if choice == 'y':
                # Attempt to repair invalid metadata files
                self.update_incomplete_minifigs()

        # Final report
        self.logger.info(f"""
Repair Summary:
--------------
Initial state:
- Images without metadata: {len(images_without_metadata)}
- Metadata without images: {len(metadata_without_images)}
- Invalid metadata files: {len(invalid_metadata)}

Final state:
- Metadata recreated: {len(images_without_metadata)}
- Orphaned metadata removed: {len(metadata_without_images) if 'choice' in locals() and choice == 'y' else 0}
- Invalid metadata repaired: {len(invalid_metadata) if 'choice' in locals() and choice == 'y' else 0}
""")

    def collect_data(self, limit=None):
        """Main data collection process"""
        try:
            # Get minifigures
            minifigs = self.get_official_minifigs(limit=limit)
            
            # Download images and save metadata
            successful_downloads = 0
            with tqdm(minifigs, desc="Downloading images") as pbar:
                for minifig in pbar:
                    if minifig.get('set_img_url'):
                        image_path = self.images_dir / f"{minifig['set_num']}.png"
                        if not image_path.exists():
                            if self.download_image(minifig['set_img_url'], image_path):
                                successful_downloads += 1
                    
                    # Save metadata
                    self.save_metadata(minifig)
                    pbar.update(1)
            
            # Print summary
            total_minifigs = len(self.get_existing_minifigs())
            years = [m['year'] for m in minifigs if isinstance(m['year'], int) and m['year'] != -1]
            year_range = f"{min(years)}-{max(years)}" if years else "unknown"
            
            self.logger.info(f"""
Dataset collection completed:
- Total minifigures: {total_minifigs}
- New minifigures added: {len(minifigs)}
- Total images downloaded: {successful_downloads}
- Year range: {year_range}
""")
            
        except Exception as e:
            self.logger.error(f"Error during data collection: {e}")
            raise

    def print_detailed_count(self):
        """Print detailed count of minifigures and images"""
        metadata_files = list(self.metadata_dir.glob('*.yaml'))
        standard_images = list(self.images_dir.glob('*.png')) + list(self.images_dir.glob('*.jpg'))
        
        custom_dir = self.images_dir / 'custom'
        custom_dirs = []
        custom_images = []
        if custom_dir.exists():
            custom_dirs = list(custom_dir.glob('custom-*'))
            for d in custom_dirs:
                custom_images.extend(list(d.glob(f'{d.name}-*.png')))
                custom_images.extend(list(d.glob(f'{d.name}-*.jpg')))

        self.logger.info(f"""
Detailed Dataset Count:
----------------------
Metadata files: {len(metadata_files)}
Standard images: {len(standard_images)}
Custom directories: {len(custom_dirs)}
Custom images: {len(custom_images)}

Total images: {len(standard_images) + len(custom_images)}

Directory Structure:
------------------
dataset/
└── images/train/
    ├── Standard images: {len(standard_images)} files
    └── custom/
        └── Custom dirs: {len(custom_dirs)} with {len(custom_images)} total images

Verification:
-----------
- Metadata without images: {sum(1 for m in metadata_files if not any(img.stem.startswith(Path(m).stem) for img in standard_images + custom_images))}
- Images without metadata: {sum(1 for img in standard_images if not (self.metadata_dir / f"{img.stem}.yaml").exists())}
""")

def main():
    try:
        print("\nChoose an option:")
        print("1. Collect new minifigures")
        print("2. Update existing minifigures with missing years")
        print("3. Both collect new and update existing")
        print("4. Validate and repair dataset")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        collector = MinifigureDataCollector()
        
        if choice == '1':
            limit_input = input("Enter number of minifigures to collect (press Enter for all): ").strip()
            limit = int(limit_input) if limit_input else None
            collector.collect_data(limit=limit)
        elif choice == '2':
            collector.update_incomplete_minifigs()
        elif choice == '3':
            limit_input = input("Enter number of minifigures to collect (press Enter for all): ").strip()
            limit = int(limit_input) if limit_input else None
            collector.collect_data(limit=limit)
            collector.update_incomplete_minifigs()
        elif choice == '4':
            collector.validate_and_repair_dataset()
        else:
            print("Invalid choice!")
            return
        
        collector.print_detailed_count()
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()