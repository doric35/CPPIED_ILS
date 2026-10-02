import os

def rewrite_with_3_decimals(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return False

    # Keep header unchanged
    header = lines[0].strip()
    new_lines = [header + "\n"]

    modified = False

    # Process matrix rows
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue

        values = stripped.split()
        try:
            floats = [float(v) for v in values]
        except ValueError:
            # Skip malformed lines
            return False

        formatted = " ".join(f"{v:.3f}" for v in floats)

        if formatted != stripped:
            modified = True

        new_lines.append(formatted + "\n")

    if not modified:
        return False

    # Overwrite file
    with open(file_path, "w") as f:
        f.writelines(new_lines)

    return True


def walk_and_rewrite(root_dir):
    modified_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        if "cppied_pod.txt" in filenames:
            file_path = os.path.join(dirpath, "cppied_pod.txt")
            if rewrite_with_3_decimals(file_path):
                modified_files.append(file_path)

    # Print modified files
    for f in modified_files:
        print(f)


if __name__ == "__main__":
    root_directory = "."  # change if needed
    walk_and_rewrite(root_directory)
