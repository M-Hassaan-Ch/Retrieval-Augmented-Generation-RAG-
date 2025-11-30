import os
import json

def collect_json_files(root_path, output_file="merged_dataset2.json"):
    merged_data = []

    # Walk through all subdirectories
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if not filename.lower().endswith(".json"):
                continue

            file_path = os.path.join(dirpath, filename)

            try:
               # Read the full JSON object
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    merged_data.append(data)

                print(f"✔ Loaded: {file_path}")

            except Exception as e:
                print(f"⚠ ERROR reading {file_path}:", e)

    # Save the final merged list
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(merged_data, out, indent=2, ensure_ascii=False)

    print("\n🎉 Merge Complete!")
    print(f"📁 Output file saved → {output_file}")


if __name__ == "__main__":
    folder = r"samples/Finished"
    collect_json_files(folder)
