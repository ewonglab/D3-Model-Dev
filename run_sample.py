import torch
import argparse
import sys
import glob
import json

from src import data
from src.load_model import load_model_local
import torch.nn.functional as F
from src import sampling
import h5py, os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from src.utils.PL_mpra import *  # Required for MPRA


def one_hot_to_seq(one_hot):
    """Convert one-hot encoded sequence back to DNA sequence."""
    # Use the correct mapping: A:0, C:1, G:2, T:3
    nucleotides = ["A", "C", "G", "T"]
    indices = np.argmax(one_hot, axis=-1)
    return "".join([nucleotides[idx] for idx in indices])


def get_cell_type_from_label(label_vector):
    """Convert one-hot encoded cell type label back to cell type name."""
    cell_types = [
        "Endothelial",
        "Fibroblast",
        "Smooth_Muscle",
        "Ventricular_Cardiomyocyte",
        "Atrial_Cardiomyocyte",
    ]
    idx = torch.argmax(label_vector).item()
    return cell_types[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from a specified dataset"
    )
    parser.add_argument(
        "--model_path", default="", type=str, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="human_heart_data.h5",
        help="Path to the input .h5 dataset file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=500, help="Batch size for processing"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of sampling steps"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="model_results/generated_results_c5_finetune_noseed",
        help="Directory to save generated results",
    )
    parser.add_argument(
        "--random_seeds",
        type=str,
        default=None,
        help="Path to random seeds JSON file. If not provided, no random seed will be used.",
    )
    parser.add_argument(
        "--save_attention",
        action="store_true",
        help="Enable saving of attention scores during sampling",
    )
    parser.add_argument(
        "--attention_steps",
        type=str,
        default="first,middle,last",
        help="Which steps to save attention for (comma-separated list or 'all')",
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory-efficient attention scoring (only last step, last layer, last head)",
    )
    args = parser.parse_args()

    # Load random seeds if provided
    random_seeds = None
    if args.random_seeds:
        try:
            with open(args.random_seeds, "r") as f:
                data = json.load(f)
                random_seeds = data["seeds"]
            print(f"Loaded {len(random_seeds)} random seeds from {args.random_seeds}")
        except Exception as e:
            print(f"Warning: Could not load random seeds: {e}")
            print("Continuing without random seeds")

    device = torch.device("cuda")
    model, graph, noise = load_model_local(args.model_path, device)

    # Load data and prepare dataset
    filepath = args.input_data
    if not os.path.exists(filepath):
        print(f"Error: Input data file {filepath} not found")
        sys.exit(1)

    data = h5py.File(filepath, "r")
    X_test = torch.tensor(np.array(data["x_test"]).astype(np.float32)).permute(0, 2, 1)
    y_test = torch.tensor(np.array(data["y_test"]).astype(np.float32))
    seq_length = X_test.shape[2]  # Get sequence length from data
    X_test = torch.argmax(X_test, dim=1)
    testing_ds = TensorDataset(X_test, y_test)
    test_ds = torch.utils.data.DataLoader(
        testing_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Initialize sampling function with batch_size
    val_pred_seq = []
    val_labels = []  # To store cell type labels
    all_attention_scores = []  # To store attention scores if enabled

    # Parse which steps to save attention for
    save_attention_steps = []
    if args.save_attention:
        if args.attention_steps == "all" and not args.memory_efficient:
            save_attention_steps = "all"
        else:
            try:
                step_names = args.attention_steps.split(",")
                save_attention_steps = step_names
            except:
                print(
                    f"Warning: Could not parse attention_steps: {args.attention_steps}"
                )
                print("Defaulting to first, middle, last steps")
                save_attention_steps = ["first", "middle", "last"]

    # Initial sampling_fn creation
    sampling_fn = sampling.get_pc_sampler(
        graph,
        noise,
        (args.batch_size, seq_length),
        "analytic",
        args.steps,
        device=device,
        save_attention=args.save_attention,
    )

    # Create generated_results directory if it doesn't exist
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    # Generate samples
    for batch_idx, (batch, val_target) in enumerate(test_ds):
        current_seed_for_this_batch = None
        if random_seeds:
            seed_idx = batch_idx % len(random_seeds)  # Cycle through available seeds
            current_seed_for_this_batch = random_seeds[seed_idx]
            print(f"Using seed {current_seed_for_this_batch} for batch {batch_idx}")

        if batch.shape[0] != args.batch_size:
            # Recreate sampling_fn if batch size changes
            sampling_fn = sampling.get_pc_sampler(
                graph,
                noise,
                (batch.shape[0], seq_length),
                "analytic",
                args.steps,
                device=device,
                save_attention=args.save_attention,
            )

        # Pass the current_seed_for_this_batch to the sampling_fn call
        if args.save_attention:
            sample, batch_attention_scores = sampling_fn(
                model, val_target.to(device), current_seed=current_seed_for_this_batch
            )
            # Store the attention scores for this batch
            all_attention_scores.append(batch_attention_scores)
        else:
            sample = sampling_fn(
                model, val_target.to(device), current_seed=current_seed_for_this_batch
            )

        seq_pred_one_hot = F.one_hot(sample, num_classes=4).float()
        val_pred_seq.append(seq_pred_one_hot)
        val_labels.append(val_target)

    val_pred_seqs = torch.cat(val_pred_seq, dim=0)
    val_labels_all = torch.cat(val_labels, dim=0)

    # Group sequences by cell type
    cell_type_groups = {}
    cell_type_attention_groups = {}

    for i, label_vector in enumerate(val_labels_all):
        cell_type = get_cell_type_from_label(label_vector)

        if cell_type not in cell_type_groups:
            cell_type_groups[cell_type] = []
            if args.save_attention and all_attention_scores:
                cell_type_attention_groups[cell_type] = []

        cell_type_groups[cell_type].append(val_pred_seqs[i])

        # Group attention scores by cell type if available
        if args.save_attention and all_attention_scores:
            # Find which batch this sequence belongs to
            batch_idx = i // args.batch_size
            if batch_idx < len(all_attention_scores):
                cell_type_attention_groups[cell_type].append(all_attention_scores[batch_idx])

    # Save sequences for each cell type separately
    for cell_type, sequences in cell_type_groups.items():
        sequences_tensor = torch.stack(sequences)

        # Create output filenames using cell type
        npz_file = f"sample_{cell_type}.npz"
        npz_path = os.path.join(results_dir, npz_file)
        txt_file = f"final_{cell_type}.txt"
        txt_path = os.path.join(results_dir, txt_file)

        # Save the generated sequences as NPZ
        np.savez(
            npz_path,
            sequences_tensor.cpu().numpy(),
        )
        print(f"Generated {len(sequences)} sequences for {cell_type} saved to {npz_path}")

        # Convert one-hot sequences to DNA sequences and save as text
        print(f"Converting {cell_type} sequences to DNA...")
        dna_sequences = []
        for seq in sequences_tensor.cpu().numpy():
            dna_seq = one_hot_to_seq(seq)
            dna_sequences.append(dna_seq)

        # Save DNA sequences to text file
        print(f"Saving {cell_type} DNA sequences to {txt_path}...")
        with open(txt_path, "w") as f:
            for seq in dna_sequences:
                f.write(f"{seq}\n")

        print(f"Successfully saved {len(dna_sequences)} {cell_type} DNA sequences to {txt_path}")
        if len(dna_sequences) > 0:
            print(f"First {cell_type} sequence: {dna_sequences[0][:50]}...")

        # Save attention scores if enabled
        if args.save_attention and cell_type in cell_type_attention_groups:
            attention_file = f"attention_{cell_type}.pt"
            attention_path = os.path.join(results_dir, attention_file)

            print(f"Saving {cell_type} attention scores to {attention_path}...")
            torch.save(cell_type_attention_groups[cell_type], attention_path)

    # Print summary
    print(f"\nSummary:")
    for cell_type, sequences in cell_type_groups.items():
        print(f"  {cell_type}: {len(sequences)} sequences")

    if args.save_attention:
        if args.memory_efficient:
            print(f"Attention scores saved in memory-efficient mode (last step, last layer only)")
        else:
            print(f"Attention scores saved successfully!")


if __name__ == "__main__":
    main()
