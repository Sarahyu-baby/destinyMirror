import os
import cv2
import pandas as pd
from face_analyzer import FaceAnalyzer
from face_visualizer import FaceVisualizer


def process_folder(root_folder_path, output_csv_path):
    """
    Traverses the folder and its subfolders, reads images, analyzes facial features,
    and saves the results to a CSV file.

    It uses the 'subfolder name' as the 'Celebrity Name'.
    It also saves the first 5 processed visualization images as examples.

    Args:
        root_folder_path (str): Root directory containing celebrity subfolders.
        output_csv_path (str): Path for the output CSV file.
    """

    # 1. Initialize analyzer and visualizer
    analyzer = FaceAnalyzer()
    visualizer = FaceVisualizer()

    # List to store all analysis results
    all_results = []

    # Configuration for sample image output
    sample_output_folder = "processed_samples"
    if not os.path.exists(sample_output_folder):
        os.makedirs(sample_output_folder)

    max_samples = 5  # Maximum number of samples to save
    sample_count = 0  # Counter for saved samples

    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    if not os.path.exists(root_folder_path):
        print(f"Error: Folder '{root_folder_path}' not found.")
        return

    print(f"Starting traversal of '{root_folder_path}'...")

    image_count = 0

    # 2. Walk through the directory tree
    for current_root, dirs, files in os.walk(root_folder_path):
        celebrity_name = os.path.basename(current_root)

        if current_root == root_folder_path:
            pass

        for filename in files:
            if filename.lower().endswith(image_extensions):
                image_count += 1
                image_path = os.path.join(current_root, filename)

                print(f"[{image_count}] Processing: {celebrity_name} - {filename}...")

                image = cv2.imread(image_path)

                if image is None:
                    print(f"  -> Warning: Could not read image '{image_path}'. Skipping.")
                    continue

                stats = analyzer.process_image(image)

                if stats:
                    # --- Visualization & Sample Saving Logic ---
                    if sample_count < max_samples:
                        print(f"  -> Generating visualization sample ({sample_count + 1}/{max_samples})...")
                        vis_img = visualizer.draw_landmarks(image, analyzer.landmarks_np)
                        vis_img = visualizer.draw_custom_points(vis_img, analyzer.custom_points)

                        sample_filename = f"sample_{sample_count + 1}_{filename}"
                        save_path = os.path.join(sample_output_folder, sample_filename)

                        cv2.imwrite(save_path, vis_img)
                        sample_count += 1
                    # -------------------------------------------

                    record = {
                        'Celebrity': celebrity_name,
                        'Filename': filename,
                    }
                    record.update(stats)
                    all_results.append(record)
                else:
                    print(f"  -> No face detected in: {filename}")

    # 3. Save to CSV (Modified)
    if all_results:
        print(f"\nAnalysis complete. Extracted {len(all_results)} records. Saving to '{output_csv_path}'...")

        # Create DataFrame
        df = pd.DataFrame(all_results)

        # Reorder columns
        cols = list(df.columns)
        if 'Celebrity' in cols: cols.remove('Celebrity')
        if 'Filename' in cols: cols.remove('Filename')
        final_cols = ['Celebrity', 'Filename'] + cols
        df = df[final_cols]

        try:
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"Success! CSV file created: {output_csv_path}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")
            print("Tip: Check if the CSV file is currently open in Excel or another program.")

        if sample_count > 0:
            print(
                f"\nGenerated {sample_count} visualization samples. Please check the folder: '{sample_output_folder}'")

    else:
        print("\nNo valid face data extracted from any images.")


if __name__ == "__main__":
    # --- Configuration ---

    input_folder = "celebrity_faces"

    # Changed extension to .csv
    output_file = "celebrity_face_features.csv"

    if os.path.exists(input_folder):
        process_folder(input_folder, output_file)
    else:
        print(f"Error: Folder '{input_folder}' not found in the current directory.")
        print("Please check the path or move the script to the parent directory of the folder.")