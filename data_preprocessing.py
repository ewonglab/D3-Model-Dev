import numpy as np
import h5py
import os
import glob
from tqdm import tqdm
import random
import re


def seq_to_one_hot(sequence):
    """
    Converts a DNA sequence string (containing A, C, G, T) into a one-hot encoded NumPy array.

    Args:
        sequence (str): The DNA sequence string (e.g., "ACGT"). Case-insensitive.

    Returns:
        np.ndarray: A 2D NumPy array of shape (L, 4) where L is the sequence length,
                    representing the one-hot encoding. Returns None if invalid characters
                    are found.
    """
    sequence = sequence.upper()
    # Mapping according to the repository's convention
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    num_bases = len(sequence)
    one_hot_encoded = np.zeros((num_bases, 4), dtype=np.float32)

    for i, base in enumerate(sequence):
        if base in mapping:
            one_hot_encoded[i, mapping[base]] = 1.0
        else:
            print(
                f"Error: Invalid character '{base}' at position {i} in sequence '{sequence}'."
            )
            return None  # Or raise an error

    return one_hot_encoded


def read_fasta(file_path):
    """Read sequences and headers from a FASTA file."""
    sequences = []
    headers = []

    with open(file_path, "r") as f:
        current_header = None
        current_seq = ""

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # If we've been building a sequence and find a new header
                if current_header and current_seq:
                    sequences.append(current_seq)
                    headers.append(current_header)

                current_header = line[1:]  # Store header without '>'
                current_seq = ""
            else:
                # Add to current sequence
                current_seq += line

        # Don't forget the last sequence
        if current_header and current_seq:
            sequences.append(current_seq)
            headers.append(current_header)

    return headers, sequences


def pad_or_truncate_sequences(sequences, target_length=200):
    """Ensure all sequences are the same length by padding or truncating."""
    processed_sequences = []

    for seq in sequences:
        if len(seq) > target_length:
            # Truncate from the center (keep both ends)
            excess = len(seq) - target_length
            start_cut = excess // 2
            processed_sequences.append(seq[:start_cut] + seq[start_cut + excess :])
        elif len(seq) < target_length:
            # Pad with N (which will be converted to zeros later)
            padding_needed = target_length - len(seq)
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            processed_sequences.append("N" * pad_left + seq + "N" * pad_right)
        else:
            processed_sequences.append(seq)

    return processed_sequences


def convert_to_one_hot(sequences, target_length=200):
    """Convert a list of sequences to one-hot encoding."""
    num_sequences = len(sequences)
    one_hot_data = np.zeros((num_sequences, target_length, 4), dtype=np.int8)

    for i, seq in tqdm(
        enumerate(sequences), total=num_sequences, desc="Converting to one-hot"
    ):
        one_hot_seq = seq_to_one_hot(seq)

        # Handle any None values (invalid sequences)
        if one_hot_seq is None:
            # Replace with zeros
            one_hot_seq = np.zeros((len(seq), 4), dtype=np.float32)

        # Ensure correct shape
        if one_hot_seq.shape[0] == target_length:
            one_hot_data[i] = one_hot_seq.astype(np.int8)
        else:
            print(
                f"Warning: Sequence {i} has unexpected length {one_hot_seq.shape[0]} (expected {target_length})"
            )
            # Try to fix by padding or truncating
            if one_hot_seq.shape[0] < target_length:
                padding = np.zeros(
                    (target_length - one_hot_seq.shape[0], 4), dtype=np.int8
                )
                one_hot_data[i] = np.vstack([one_hot_seq.astype(np.int8), padding])
            else:
                one_hot_data[i] = one_hot_seq[:target_length].astype(np.int8)

    return one_hot_data


def main():
    # Define paths
    data_dir = "generated_c5_human_heart_seqs"
    output_file = "generated_c5_human_heart_data.h5"

    # Define cell types and create mapping
    cell_types = [
        "Endothelial",
        "Fibroblast",
        "Smooth_Muscle",
        "Ventricular_Cardiomyocyte",
        "Atrial_Cardiomyocyte",
    ]
    cell_type_to_idx = {cell_type: i for i, cell_type in enumerate(cell_types)}

    # Initialize lists to store data
    train_sequences = []
    train_cell_types = []
    valid_sequences = []
    valid_cell_types = []
    test_sequences = []
    test_cell_types = []

    # Process training files
    train_files = glob.glob(os.path.join(data_dir, "*train_*.fasta"))
    for file_path in train_files:
        # Extract cell type from filename
        cell_type = re.search(r"train_(.+?)\.fasta", os.path.basename(file_path)).group(
            1
        )

        # Read sequences
        headers, sequences = read_fasta(file_path)

        # Add to lists
        train_sequences.extend(sequences)
        train_cell_types.extend([cell_type] * len(sequences))

    # Process validation files
    valid_files = glob.glob(os.path.join(data_dir, "*val_*.fasta"))
    for file_path in valid_files:
        # Extract cell type from filename
        cell_type = re.search(r"val_(.+?)\.fasta", os.path.basename(file_path)).group(1)

        # Read sequences
        headers, sequences = read_fasta(file_path)

        # Add to lists
        valid_sequences.extend(sequences)
        valid_cell_types.extend([cell_type] * len(sequences))

    # Process test files
    test_files = glob.glob(os.path.join(data_dir, "test_*.fasta"))
    for file_path in test_files:
        # Extract cell type from filename
        cell_type = re.search(r"test_(.+?)\.fa", os.path.basename(file_path)).group(1)

        # Read sequences
        headers, sequences = read_fasta(file_path)

        # Add to lists
        test_sequences.extend(sequences)
        test_cell_types.extend([cell_type] * len(sequences))

    print(
        f"Loaded {len(train_sequences)} training sequences, {len(valid_sequences)} validation sequences, and {len(test_sequences)} test sequences"
    )

    # Ensure sequences are all the same length
    target_length = 200
    train_sequences = pad_or_truncate_sequences(train_sequences, target_length)
    valid_sequences = pad_or_truncate_sequences(valid_sequences, target_length)
    test_sequences = pad_or_truncate_sequences(test_sequences, target_length)

    # Convert sequences to one-hot encoding
    print("Converting training sequences to one-hot...")
    x_train = convert_to_one_hot(train_sequences, target_length)
    print("Converting validation sequences to one-hot...")
    x_valid = convert_to_one_hot(valid_sequences, target_length)
    print("Converting test sequences to one-hot...")
    x_test = convert_to_one_hot(test_sequences, target_length)

    # Create one-hot encoded activity vectors (y)
    y_train = np.zeros((len(train_cell_types), len(cell_types)), dtype=np.float32)
    for i, cell_type in enumerate(train_cell_types):
        y_train[i, cell_type_to_idx[cell_type]] = 1.0

    y_valid = np.zeros((len(valid_cell_types), len(cell_types)), dtype=np.float32)
    for i, cell_type in enumerate(valid_cell_types):
        y_valid[i, cell_type_to_idx[cell_type]] = 1.0

    y_test = np.zeros((len(test_cell_types), len(cell_types)), dtype=np.float32)
    for i, cell_type in enumerate(test_cell_types):
        y_test[i, cell_type_to_idx[cell_type]] = 1.0

    print(f"Final dataset sizes:")
    print(f"  Training: {x_train.shape[0]} sequences")
    print(f"  Validation: {x_valid.shape[0]} sequences")
    print(f"  Test: {x_test.shape[0]} sequences")

    # Save to H5 file
    with h5py.File(output_file, "w") as f:
        f.create_dataset("x_train", data=x_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("x_valid", data=x_valid)
        f.create_dataset("y_valid", data=y_valid)
        f.create_dataset("x_test", data=x_test)
        f.create_dataset("y_test", data=y_test)

    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    main()