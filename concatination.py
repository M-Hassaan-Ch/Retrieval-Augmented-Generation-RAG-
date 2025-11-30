import json

# Input file names
file1 = "merged_dataset.json"
file2 = "merged_dataset2.json"

# Output file name
output_file = "merged_final.json"

def read_json_list(filepath):
    """Safely read a JSON list from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{filepath} does not contain a JSON list.")
    return data


def main():
    data1 = read_json_list(file1)
    data2 = read_json_list(file2)

    # Concatenate lists
    merged = data1 + data2

    # Write to new file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Successfully merged {len(data1)} + {len(data2)} entries.")
    print(f"Saved final dataset to: {output_file}")


if __name__ == "__main__":
    main()
