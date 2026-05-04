import os
import shutil
import argparse

def collect_csv_files(source_dir, destination_dir, target_filename_base):
    """
    Finds CSV files starting with a specific base name in subdirectories, 
    copies them to a destination folder, and renames them by combining 
    their parent directory's name with the original file's unique ID suffix.
    """
    # --- 1. Validate source directory ---
    if not os.path.isdir(source_dir):
        print(f"❌ Error: The source directory '{source_dir}' does not exist.")
        return

    # --- 2. Create the destination directory ---
    os.makedirs(destination_dir, exist_ok=True)
    print(f"✅ Scanning '{source_dir}' for files starting with '{target_filename_base}*.csv'...")
    print(f"✅ Results will be saved in '{destination_dir}'.")

    copied_files_count = 0
    found_in_folders = 0

    # --- 3. Scan all items in the source directory ---
    for item_name in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, item_name)

        # Process only if the item is a directory
        if os.path.isdir(subfolder_path):
            files_found_in_subdir = False
            # --- 4. Scan all files within the subfolder ---
            for filename in os.listdir(subfolder_path):
                # Check if the file matches the pattern (e.g., starts with 'program_output' and ends with '.csv')
                if filename.startswith(target_filename_base) and filename.endswith('.csv'):
                    # --- 5. Extract the suffix from the original filename ---
                    # This captures everything between the base name and the '.csv' extension.
                    # e.g., for 'program_output_1.csv', it extracts '_1'
                    # e.g., for 'program_output.csv', it extracts '' (an empty string)
                    file_suffix = filename[len(target_filename_base):-4] # -4 to remove '.csv'
                    
                    # --- 6. Define new name and copy the file ---
                    new_filename = f"{item_name}{file_suffix}.csv"
                    source_file = os.path.join(subfolder_path, filename)
                    destination_file = os.path.join(destination_dir, new_filename)
                    
                    try:
                        shutil.copy(source_file, destination_file)
                        if not files_found_in_subdir:
                             print(f"  📂 In folder '{item_name}':")
                             files_found_in_subdir = True
                        print(f"    -> Copied: {filename}  ->  {new_filename}")
                        copied_files_count += 1
                    except Exception as e:
                        print(f"    ❌ Error copying file {filename}: {e}")
            
            if files_found_in_subdir:
                found_in_folders += 1


    print(f"\n✨ Done! Found and copied a total of {copied_files_count} file(s) from {found_in_folders} folder(s).")

if __name__ == '__main__':
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Collects specific CSV files from subfolders into a single output directory."
    )
    
    parser.add_argument(
        '--source_dir', 
        type=str,
        required=True,
        help="The parent directory containing the subfolders to scan (e.g., 'all_my_results')."
    )
    
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default='evaluation',
        help="The name of the directory to save the copied files. Defaults to 'evaluation'."
    )
    
    parser.add_argument(
        '-f', '--filename_base', 
        type=str, 
        default='program_output',
        help="The base name of the CSV files to look for. Defaults to 'program_output'."
    )
    
    args = parser.parse_args()
    
    # --- Run the main function with the provided arguments ---
    collect_csv_files(args.source_dir, args.output, args.filename_base)