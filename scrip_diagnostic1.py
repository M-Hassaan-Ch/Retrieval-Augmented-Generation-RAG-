import os
import json

def merge_all_json(input_folder: str, output_file: str = "merged_dataset.json"):
    merged_data = []

    # Loop through folder
    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".json"):
            continue  # skip non-json files

        file_path = os.path.join(input_folder, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Append the entire JSON structure
                merged_data.append(data)

        except Exception as e:
            print(f"⚠️ Error reading {file_name}: {e}")

    # Write final collected data
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(merged_data, out, indent=2, ensure_ascii=False)

    print(f"✅ Merged dataset created → {output_file}")


if __name__ == "__main__":
    folder = r"diagnostic_kg/Diagnosis_flowchart"
    merge_all_json(folder)
