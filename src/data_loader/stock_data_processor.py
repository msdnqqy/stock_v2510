import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class StockDataProcessor:
    def __init__(self, input_dir='/mnt/d/projects/stock_v2510/data/stock', output_dir='/mnt/d/projects/stock_v2510/data/stock_pre'):
        """
        Initialize the StockDataProcessor.

        Args:
            input_dir (str): Directory containing the input CSV files
            output_dir (str): Directory to save processed files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.new_columns = ['date', 'open', 'high', 'low', 'close', 'vol', 'amount']

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_file(self, file_path):
        """
        Process a single CSV file.

        Args:
            file_path (Path): Path to the input CSV file

        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Read the CSV file without header
            df = pd.read_csv(file_path, header=None, encoding='gbk')

            # Skip if file is empty or has only one line (header)
            if len(df) <= 1:
                return False

            # Remove the first row (original header) and last row
            df = df.iloc[:-1]

            # Set new column names
            df.columns = self.new_columns

            # Save to output directory with the same filename
            output_path = self.output_dir / file_path.name
            df.to_csv(output_path, index=False, encoding='utf-8')

            return True

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False

    def process_all_files(self):
        """Process all CSV files in the input directory."""
        # Get all CSV files in the input directory
        csv_files = list(self.input_dir.glob('*.csv'))

        if not csv_files:
            print(f"No CSV files found in {self.input_dir}")
            return

        print(f"Found {len(csv_files)} files to process...")

        # Process files with progress bar
        success_count = 0
        for file_path in tqdm(csv_files, desc="Processing files"):
            if self.process_file(file_path):
                success_count += 1

        print(f"\nProcessing complete. Successfully processed {success_count}/{len(csv_files)} files.")
        print(f"Processed files are saved in: {self.output_dir}")


if __name__ == "__main__":
    # Example usage
    processor = StockDataProcessor()
    processor.process_all_files()

    processor = StockDataProcessor(input_dir='/mnt/d/projects/stock_v2510/data/industry', output_dir='/mnt/d/projects/stock_v2510/data/industry_pre')
    processor.process_all_files()
