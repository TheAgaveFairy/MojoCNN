#!/usr/bin/env python3
"""
Compare weights/biases from a text file with binary data from a .dat file.
Usage: python compare_weights.py weights.txt model_f32.dat
       python compare_weights.py weights.txt model_f64.dat
"""

import sys
import struct
import numpy as np
from pathlib import Path


def parse_text_file(text_file):
    """Parse text file and extract all numerical values."""
    numbers = []

    with open(text_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Skip empty lines and comments
            line = line.strip()
            # print(line)
            if not line or line.startswith('#') or line.startswith('//'):
                continue

            # Split line into tokens and try to parse each as a number
            tokens = line.split(",")
            for token in tokens:
                try:
                    # Try to parse as float
                    num = float(token)
                    numbers.append(num)
                except ValueError:
                    # Skip non-numeric tokens (could be labels, etc.)
                    continue

    print(f"Parsed {len(numbers)} numbers from text file")
    return numbers


def read_binary_file(binary_file):
    """Read binary file and determine data type from filename."""

    # Determine data type from filename
    filename = Path(binary_file).name.lower()
    if 'f32' in filename or 'float32' in filename:
        dtype = 'f'  # 4-byte float
        np_dtype = np.float32
        bytes_per_num = 4
        print("Detected f32 (float32) format")
    elif 'f64' in filename or 'float64' in filename or 'double' in filename:
        dtype = 'd'  # 8-byte double
        np_dtype = np.float64
        bytes_per_num = 8
        print("Detected f64 (float64) format")
    else:
        # Default guess based on file size if we can't determine from name
        file_size = Path(binary_file).stat().st_size
        print(f"Binary file size: {file_size} bytes")

        # Try f32 first
        if file_size % 4 == 0:
            dtype = 'f'
            np_dtype = np.float32
            bytes_per_num = 4
            print("Guessing f32 format (file size divisible by 4)")
        elif file_size % 8 == 0:
            dtype = 'd'
            np_dtype = np.float64
            bytes_per_num = 8
            print("Guessing f64 format (file size divisible by 8)")
        else:
            raise ValueError(
                "Cannot determine data type and file size doesn't match f32 or f64")

    # Read binary data
    numbers = []
    with open(binary_file, 'rb') as f:
        while True:
            data = f.read(bytes_per_num)
            if not data:
                break
            if len(data) != bytes_per_num:
                print(f"Warning: Incomplete read at end of file (got {
                      len(data)} bytes, expected {bytes_per_num})")
                break

            # Unpack binary data
            num = struct.unpack('<' + dtype, data)[0]  # Little-endian
            numbers.append(float(num))

    print(f"Read {len(numbers)} numbers from binary file")
    return numbers, np_dtype


def compare_arrays(text_numbers, binary_numbers, tolerance_abs=1e-6, tolerance_rel=1e-6):
    """Compare two arrays of numbers with detailed statistics."""

    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Text file numbers: {len(text_numbers)}")
    print(f"Binary file numbers: {len(binary_numbers)}")

    if len(text_numbers) != len(binary_numbers):
        print(f"❌ LENGTH MISMATCH!")
        print(f"Text has {len(text_numbers)} numbers, binary has {
              len(binary_numbers)}")

        # Compare what we can
        min_len = min(len(text_numbers), len(binary_numbers))
        text_numbers = text_numbers[:min_len]
        binary_numbers = binary_numbers[:min_len]
        print(f"Comparing first {min_len} numbers...")

    # Convert to numpy arrays for easier comparison
    text_arr = np.array(text_numbers, dtype=np.float64)
    binary_arr = np.array(binary_numbers, dtype=np.float64)

    # Calculate differences
    abs_diff = np.abs(text_arr - binary_arr)
    rel_diff = np.abs(abs_diff / (np.abs(text_arr) + 1e-15)
                      )  # Avoid division by zero

    # Statistics
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    mean_rel_diff = np.mean(rel_diff)

    # Find matches within tolerance
    abs_matches = abs_diff <= tolerance_abs
    rel_matches = rel_diff <= tolerance_rel
    close_matches = abs_matches | rel_matches

    num_exact = np.sum(text_arr == binary_arr)
    num_close = np.sum(close_matches)
    num_different = len(text_arr) - num_close

    print(f"\n--- STATISTICS ---")
    print(f"Exact matches: {num_exact}/{len(text_arr)
                                        } ({100*num_exact/len(text_arr):.2f}%)")
    print(f"Close matches (within tolerance): {
          num_close}/{len(text_arr)} ({100*num_close/len(text_arr):.2f}%)")
    print(f"Significant differences: {
          num_different}/{len(text_arr)} ({100*num_different/len(text_arr):.2f}%)")

    print(f"\n--- DIFFERENCE METRICS ---")
    print(f"Max absolute difference: {max_abs_diff:.2e}")
    print(f"Max relative difference: {max_rel_diff:.2e}")
    print(f"Mean absolute difference: {mean_abs_diff:.2e}")
    print(f"Mean relative difference: {mean_rel_diff:.2e}")

    # Show some examples
    print(f"\n--- FIRST 10 COMPARISONS ---")
    for i in range(min(10, len(text_arr))):
        status = "✓" if close_matches[i] else "❌"
        print(f"{i:3d}: {status} Text: {text_arr[i]:12.6e} | Binary: {
              binary_arr[i]:12.6e} | Diff: {abs_diff[i]:8.2e}")

    # Show worst mismatches
    if num_different > 0:
        print(f"\n--- WORST 5 MISMATCHES ---")
        # Top 5 largest differences
        worst_indices = np.argsort(abs_diff)[-5:][::-1]
        for idx in worst_indices:
            if not close_matches[idx]:  # Only show actual mismatches
                print(f"{idx:3d}: ❌ Text: {text_arr[idx]:12.6e} | Binary: {
                      binary_arr[idx]:12.6e} | Diff: {abs_diff[idx]:8.2e}")

    # Overall assessment
    print(f"\n--- ASSESSMENT ---")
    if num_exact == len(text_arr):
        print("🎉 PERFECT MATCH! All numbers are exactly identical.")
    elif num_close == len(text_arr):
        print("✅ CLOSE MATCH! All numbers are within tolerance.")
    elif num_close / len(text_arr) > 0.95:
        print("⚠️  MOSTLY GOOD! Most numbers match, but some significant differences.")
    else:
        print("❌ POOR MATCH! Many numbers differ significantly.")

    return {
        'exact_matches': num_exact,
        'close_matches': num_close,
        'total_numbers': len(text_arr),
        'max_abs_diff': max_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_abs_diff': mean_abs_diff,
        'mean_rel_diff': mean_rel_diff
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_weights.py <text_file> <binary_file>")
        print("Example: python compare_weights.py weights.txt model_f32.dat")
        sys.exit(1)

    text_file = sys.argv[1]
    binary_file = sys.argv[2]

    # Check if files exist
    if not Path(text_file).exists():
        print(f"Error: Text file '{text_file}' not found!")
        sys.exit(1)

    if not Path(binary_file).exists():
        print(f"Error: Binary file '{binary_file}' not found!")
        sys.exit(1)

    print(f"Comparing:")
    print(f"  Text file: {text_file}")
    print(f"  Binary file: {binary_file}")
    print()

    try:
        # Parse files
        text_numbers = parse_text_file(text_file)
        binary_numbers, binary_dtype = read_binary_file(binary_file)

        if len(text_numbers) == 0:
            print("Error: No numbers found in text file!")
            sys.exit(1)

        if len(binary_numbers) == 0:
            print("Error: No numbers found in binary file!")
            sys.exit(1)

        # Set tolerances based on data type
        if binary_dtype == np.float32:
            tolerance_abs = 1e-6
            tolerance_rel = 1e-6
            print(f"Using f32 tolerances: abs={
                  tolerance_abs:.0e}, rel={tolerance_rel:.0e}")
        else:
            tolerance_abs = 1e-12
            tolerance_rel = 1e-12
            print(f"Using f64 tolerances: abs={
                  tolerance_abs:.0e}, rel={tolerance_rel:.0e}")

        # Compare
        results = compare_arrays(
            text_numbers, binary_numbers, tolerance_abs, tolerance_rel)

        # Exit code based on results
        if results['close_matches'] == results['total_numbers']:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Differences found

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
