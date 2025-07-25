#!/usr/bin/env python3
import subprocess
import sys
from typing import List, Tuple


def run_mojo() -> List[float]:
    """Run the Mojo program and parse float output from stdout."""
    try:
        result = subprocess.run(
            ["mojo", "lenetgpu.mojo"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print(f"Error running mojo: {result.stderr}")
            return []

        # Parse comma-separated floats from stdout
        output = result.stdout.strip()
        if not output:
            return []

        # Handle multiple lines - concatenate all comma-separated values
        all_values = []
        for line in output.split('\n'):
            if line.strip():
                values = [float(x.strip())
                          for x in line.split(',') if x.strip()]
                all_values.extend(values)

        return all_values

    except subprocess.TimeoutExpired:
        print("Timeout running mojo program")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []


def compare_outputs(outputs: List[List[float]], tolerance: float = 1e-6) -> None:
    """Compare multiple outputs and report differences."""
    if len(outputs) < 2:
        print("Need at least 2 runs to compare")
        return

    # Check if all outputs have same length
    lengths = [len(out) for out in outputs]
    if len(set(lengths)) > 1:
        print(f"WARNING: Output lengths differ: {lengths}")
        min_len = min(lengths)
        print(f"Comparing first {min_len} values only")
    else:
        min_len = lengths[0]
        print(f"All outputs have {min_len} values")

    # Compare first output with all others
    reference = outputs[0][:min_len]
    differences_found = False

    for run_idx in range(1, len(outputs)):
        current = outputs[run_idx][:min_len]

        print(f"\n=== Comparing Run 1 vs Run {run_idx + 1} ===")
        run_differences = []

        for pos in range(min_len):
            diff = abs(reference[pos] - current[pos])
            if diff > tolerance:
                run_differences.append(
                    (pos, reference[pos], current[pos], diff))

        if run_differences:
            differences_found = True
            print(f"Found {len(run_differences)} differences:")
            # Show first 10
            for pos, ref_val, cur_val, diff in run_differences[:10]:
                print(f"  Position {pos}: {ref_val} vs {
                      cur_val} (diff: {diff:.2e})")
            if len(run_differences) > 10:
                print(f"  ... and {
                      len(run_differences) - 10} more differences")
        else:
            print("No differences found (within tolerance)")

    if not differences_found:
        print(
            f"\n✅ All runs produced identical results (tolerance: {tolerance})")
    else:
        print(f"\n⚠️  Differences detected between runs!")


def main():
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-6

    print(f"Running 'mojo lenetgpu.mojo' {num_runs} times...")
    print(f"Tolerance for differences: {tolerance}")

    outputs = []
    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}...", end=" ", flush=True)
        output = run_mojo()
        if output:
            outputs.append(output)
            print(f"Got {len(output)} values")
        else:
            print("Failed!")

    if len(outputs) == 0:
        print("No successful runs!")
        return

    print(f"\nSuccessfully completed {len(outputs)} runs")

    # Show sample of first run's output
    if outputs[0]:
        print(f"\nSample from first run (first 10 values):")
        print(", ".join(f"{x:.6f}" for x in outputs[0][:10]))
        if len(outputs[0]) > 10:
            print("...")

    # Compare all outputs
    compare_outputs(outputs, tolerance)


if __name__ == "__main__":
    main()  # !/usr/bin/env python3
