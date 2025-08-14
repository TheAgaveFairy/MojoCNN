#!/usr/bin/env python3
import re
import sys


def parse_file(filename):
    gpu_values = {}
    cpu_values = []
    current_section = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Check for section headers
            if line.startswith('GPU'):
                current_section = 'GPU'
                gpu_label = line.split('GPU')[1]
                if not gpu_label:
                    gpu_label = "unlabeled"
                print(f"Found GPU section: {gpu_label}")
                continue
            elif line.startswith('CPU'):
                current_section = 'CPU'
                cpu_label = line.split('CPU')[1]
                if not cpu_label:
                    cpu_label = "unlabeled"
                print(f"Found CPU section: {cpu_label}")
                continue

            # Parse data lines
            if current_section == 'GPU':
                # GPU format: "index<delimiter>float"
                # Handle various delimiters (space, comma, tab, colon, etc.)
                parts = re.split(r'[,\s:;|]+', line, 1)
                if len(parts) >= 2:
                    try:
                        index = int(parts[0])
                        value = float(parts[1])
                        gpu_values[index] = value
                    except ValueError:
                        print(f"Warning: Could not parse GPU line: {line}")

            elif current_section == 'CPU':
                # CPU format: just floats with various delimiters
                # Split on common delimiters and take first valid float
                parts = re.split(r'[,\s:;|]+', line)
                for part in parts:
                    try:
                        value = float(part)
                        cpu_values.append(value)
                        break  # Only take first valid float per line
                    except ValueError:
                        continue

    return gpu_values, cpu_values


def compare_values(gpu_dict, cpu_list):
    # Sort GPU values by index
    if not gpu_dict:
        print("No GPU values found!")
        return

    gpu_sorted = [gpu_dict[i] for i in sorted(gpu_dict.keys())]

    print(f"\nGPU values: {len(gpu_sorted)} items")
    print(f"CPU values: {len(cpu_list)} items")

    if len(gpu_sorted) != len(cpu_list):
        print(f"WARNING: Length mismatch! GPU has {
              len(gpu_sorted)}, CPU has {len(cpu_list)}")
        min_len = min(len(gpu_sorted), len(cpu_list))
        print(f"Comparing first {min_len} values...")
    else:
        min_len = len(gpu_sorted)

    matches = 0
    mismatches = []

    for i in range(min_len):
        gpu_val = gpu_sorted[i]
        cpu_val = cpu_list[i]

        if gpu_val == cpu_val:
            matches += 1
        else:
            mismatches.append((i, gpu_val, cpu_val))

    print(f"\nResults:")
    print(f"Matches: {matches}/{min_len}")
    print(f"Mismatches: {len(mismatches)}")

    if mismatches:
        print(f"\nFirst few mismatches:")
        for i, (idx, gpu_val, cpu_val) in enumerate(mismatches[:10]):
            print(f"  Index {idx}: GPU={gpu_val}, CPU={cpu_val}")

        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    else:
        print("All values match exactly!")


def main():
    if len(sys.argv) != 2:
        print("Usage: python compare.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        gpu_values, cpu_values = parse_file(filename)
        compare_values(gpu_values, cpu_values)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()  # !/usr/bin/env python3
