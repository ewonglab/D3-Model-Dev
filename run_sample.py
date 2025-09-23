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
        "--attention_layer",
        type=int,
        default=None,
        help="Specify which attention layer to extract scores from (0-indexed). Default is the last layer.",
    )
    parser.add_argument(
        "--save_pca_features",
        action="store_true",
        help="Enable PCA-based feature extraction from transformer layers",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=5,
        help="Number of PCA components to extract (default: 5)",
    )
    parser.add_argument(
        "--save_concept_attention",
        action="store_true",
        help="Enable concept attention analysis during sampling",
    )
    parser.add_argument(
        "--concept_attention_layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices for concept attention (e.g., '8, 9,10,11'). Default is last 4 layers.",
    )
    parser.add_argument(
        "--concept_attention_type",
        type=str,
        default="specific",
        choices=["specific", "all"],
        help="Type of concept attention to save: 'specific' for target cell type only, 'all' for all cell types (default: specific)",
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

    # Enable concept attention if requested
    if args.save_concept_attention:
        concept_layers = None
        if args.concept_attention_layers:
            try:
                concept_layers = [int(x.strip()) for x in args.concept_attention_layers.split(',')]
                print(f"Using concept attention layers: {concept_layers}")
            except ValueError:
                print(f"Warning: Invalid concept attention layers format. Using default.")
                concept_layers = None

        model.enable_concept_attention(True, layers=concept_layers)
        print(f"🧬 Concept attention enabled with type: {args.concept_attention_type}")
    else:
        model.enable_concept_attention(False)

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

    # Adjust batch size to dataset size if dataset is smaller
    dataset_size = X_test.shape[0]
    effective_batch_size = min(args.batch_size, dataset_size)

    print(f"Dataset size: {dataset_size}, Requested batch size: {args.batch_size}")
    if effective_batch_size < args.batch_size:
        print(f"Adjusting batch size to {effective_batch_size} to match dataset size")

    testing_ds = TensorDataset(X_test, y_test)
    test_ds = torch.utils.data.DataLoader(
        testing_ds, batch_size=effective_batch_size, shuffle=False, num_workers=4
    )

    # Initialize sampling function with batch_size
    val_pred_seq = []
    val_labels = []  # To store cell type labels
    all_attention_scores = []  # To store attention scores if enabled
    all_pca_scores = [] if args.save_pca_features else None  # To store PCA scores if enabled
    all_concept_attention = [] if args.save_concept_attention else None  # To store concept attention if enabled

    # Initial sampling_fn creation
    sampling_fn = sampling.get_pc_sampler(
        graph,
        noise,
        (effective_batch_size, seq_length),
        "analytic",
        args.steps,
        device=device,
        save_attention=args.save_attention,
        attention_layer_idx=args.attention_layer,
        save_pca_features=args.save_pca_features,
        pca_components=args.pca_components,
        save_concept_attention=args.save_concept_attention,
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

        if batch.shape[0] != effective_batch_size:
            # Recreate sampling_fn if batch size changes
            sampling_fn = sampling.get_pc_sampler(
                graph,
                noise,
                (batch.shape[0], seq_length),
                "analytic",
                args.steps,
                device=device,
                save_attention=args.save_attention,
                attention_layer_idx=args.attention_layer,
                save_pca_features=args.save_pca_features,
                pca_components=args.pca_components,
                save_concept_attention=args.save_concept_attention,
            )

        # Handle different return types from the sampling function
        result = sampling_fn(model, val_target.to(device), current_seed=current_seed_for_this_batch)

        # Handle multiple return types based on enabled features
        if args.save_attention and args.save_pca_features and args.save_concept_attention:
            sample, batch_attention_scores, batch_pca_scores, batch_concept_attention = result
            all_attention_scores.append(batch_attention_scores)
            all_pca_scores.append(batch_pca_scores)
            all_concept_attention.append(batch_concept_attention)
        elif args.save_attention and args.save_pca_features:
            sample, batch_attention_scores, batch_pca_scores = result
            all_attention_scores.append(batch_attention_scores)
            all_pca_scores.append(batch_pca_scores)
        elif args.save_attention and args.save_concept_attention:
            sample, batch_attention_scores, batch_concept_attention = result
            all_attention_scores.append(batch_attention_scores)
            all_concept_attention.append(batch_concept_attention)
        elif args.save_pca_features and args.save_concept_attention:
            sample, batch_pca_scores, batch_concept_attention = result
            all_pca_scores.append(batch_pca_scores)
            all_concept_attention.append(batch_concept_attention)
        elif args.save_attention:
            sample, batch_attention_scores = result
            all_attention_scores.append(batch_attention_scores)
        elif args.save_pca_features:
            sample, batch_pca_scores = result
            all_pca_scores.append(batch_pca_scores)
        elif args.save_concept_attention:
            sample, batch_concept_attention = result
            all_concept_attention.append(batch_concept_attention)
        else:
            sample = result

        seq_pred_one_hot = F.one_hot(sample, num_classes=4).float()
        val_pred_seq.append(seq_pred_one_hot)
        val_labels.append(val_target)

    val_pred_seqs = torch.cat(val_pred_seq, dim=0)
    val_labels_all = torch.cat(val_labels, dim=0)

    # Group sequences by cell type
    cell_type_groups = {}
    cell_type_attention_groups = {}
    cell_type_pca_groups = {}
    cell_type_concept_attention_groups = {}

    for i, label_vector in enumerate(val_labels_all):
        cell_type = get_cell_type_from_label(label_vector)

        if cell_type not in cell_type_groups:
            cell_type_groups[cell_type] = []
            if args.save_attention and all_attention_scores:
                cell_type_attention_groups[cell_type] = []
            if args.save_pca_features and all_pca_scores:
                cell_type_pca_groups[cell_type] = []
            if args.save_concept_attention and all_concept_attention:
                cell_type_concept_attention_groups[cell_type] = []

        cell_type_groups[cell_type].append(val_pred_seqs[i])

        # Group attention scores by cell type if available
        if args.save_attention and all_attention_scores:
            # Find which batch this sequence belongs to
            batch_idx = i // effective_batch_size
            if batch_idx < len(all_attention_scores):
                cell_type_attention_groups[cell_type].append(all_attention_scores[batch_idx])

        # Group PCA scores by cell type if available
        if args.save_pca_features and all_pca_scores:
            # Find which batch this sequence belongs to
            batch_idx = i // effective_batch_size
            if batch_idx < len(all_pca_scores):
                cell_type_pca_groups[cell_type].append(all_pca_scores[batch_idx])

        # Group concept attention by cell type if available
        if args.save_concept_attention and all_concept_attention:
            # Find which batch this sequence belongs to
            batch_idx = i // effective_batch_size
            if batch_idx < len(all_concept_attention):
                # Store the global sequence index and batch data for proper mapping later
                cell_type_concept_attention_groups[cell_type].append({
                    'global_seq_idx': i,
                    'global_batch_idx': batch_idx,
                    'within_batch_idx': i % effective_batch_size,
                    'concept_data': all_concept_attention[batch_idx]
                })

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

        # Save PCA scores if enabled
        if args.save_pca_features and cell_type in cell_type_pca_groups:
            pca_file = f"pca_features_{cell_type}.pt"
            pca_path = os.path.join(results_dir, pca_file)

            print(f"Saving {cell_type} PCA features to {pca_path}...")
            torch.save(cell_type_pca_groups[cell_type], pca_path)

        # Save concept attention if enabled
        if args.save_concept_attention and cell_type in cell_type_concept_attention_groups:
            concept_file = f"concept_attention_{cell_type}.pt"
            concept_path = os.path.join(results_dir, concept_file)

            # Extract just the concept data for saving
            concept_data = [entry['concept_data'] for entry in cell_type_concept_attention_groups[cell_type]]
            print(f"Saving {cell_type} concept attention to {concept_path}...")
            torch.save(concept_data, concept_path)

            # Also save a detailed analysis file with extracted attention scores
            if cell_type_concept_attention_groups[cell_type]:
                concept_analysis_file = f"concept_attention_analysis_{cell_type}.json"
                concept_analysis_path = os.path.join(results_dir, concept_analysis_file)

                analysis_data = []
                for seq_idx, dna_seq in enumerate(dna_sequences):
                    # Find the corresponding concept attention data for this sequence
                    # Each entry in the concept attention group corresponds to this sequence
                    if seq_idx < len(cell_type_concept_attention_groups[cell_type]):
                        entry = cell_type_concept_attention_groups[cell_type][seq_idx]
                        concept_maps = entry['concept_data']
                        within_batch_idx = entry['within_batch_idx']

                        if concept_maps:
                            seq_concept_scores = model.extract_concept_attention_for_sequence(
                                concept_maps, within_batch_idx, cell_type, args.concept_attention_type
                            )

                            # Only add to analysis if we got valid concept scores
                            if seq_concept_scores:
                                analysis_data.append({
                                    'sequence_id': f"{cell_type}_seq_{seq_idx:04d}",
                                    'cell_type': cell_type,
                                    'sequence': dna_seq,
                                    'concept_attention_scores': seq_concept_scores,
                                    'attention_type': args.concept_attention_type
                                })

                if analysis_data:
                    import json
                    with open(concept_analysis_path, 'w') as f:
                        json.dump(analysis_data, f, indent=2)
                    print(f"Saving {cell_type} concept attention analysis to {concept_analysis_path}...")
                    print(f"  Analyzed {len(analysis_data)} sequences with concept attention type: {args.concept_attention_type}")

    # Print summary
    print(f"\nSummary:")
    for cell_type, sequences in cell_type_groups.items():
        print(f"  {cell_type}: {len(sequences)} sequences")

    if args.save_attention:
        if args.attention_layer is not None:
            print(f"Attention scores saved from layer {args.attention_layer}")
        else:
            print(f"Attention scores saved from the last layer")

    if args.save_pca_features:
        print(f"PCA features saved with {args.pca_components} components per position")

    if args.save_concept_attention:
        layers_used = args.concept_attention_layers if args.concept_attention_layers else "default (last 4)"
        print(f"Concept attention saved from layers: {layers_used}")
        print(f"Concept attention type: {args.concept_attention_type}")
        print(f"Output includes:")
        print(f"  - concept_attention_{{cell_type}}.pt: Raw attention maps")
        print(f"  - concept_attention_analysis_{{cell_type}}.json: Processed attention scores per sequence")


if __name__ == "__main__":
    main()
