"""
Baseline NCA Training on ARC Task 00d62c1b
===========================================
Goal: Replicate the overfitting behavior observed by Xu et al. (2025)
Expected: Model solves training examples but fails on test case
"""

import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

# Import from Guichard repo (make sure these files are in your directory)
from NCA import CA
import arc_agi_utils as aau
import utils
import vft

# Configuration
DEVICE = vft.DEVICE
CHANNELS = vft.CHANNELS
BATCH_SIZE = vft.BATCH_SIZE
GENESIZE = vft.GENESIZE

# Task-specific settings
TASK_ID = "00d62c1b"
TRAINING_ITERATIONS = 3000
LEARNING_RATE = 1e-3
STEPS_BETWEEN_ITERATIONS = (32, 64)  # Random range

# Paths
DATA_ROOT = Path("ArcData/data")
TRAINING_PATH = DATA_ROOT / "training"
OUTPUT_DIR = Path("results") / TASK_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_single_task(task_id):
    """Load a single ARC task by ID

    Each task JSON contains:
    - 'train': Training examples (what the model learns from)
    - 'test': Test examples (what we evaluate generalization on)
    """
    task_file = TRAINING_PATH / f"{task_id}.json"

    with open(task_file) as f:
        task_data = json.load(f)

    # Extract training examples
    train_inputs = [np.array(ex['input']) for ex in task_data['train']]
    train_outputs = [np.array(ex['output']) for ex in task_data['train']]

    # Extract test examples
    test_inputs = [np.array(ex['input']) for ex in task_data['test']]
    test_outputs = [np.array(ex['output']) for ex in task_data['test']]

    return train_inputs, train_outputs, test_inputs, test_outputs


def visualize_arc_task(inputs, outputs, title="ARC Task"):
    """Visualize input/output pairs"""
    n = len(inputs)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4*n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        axes[i][0].imshow(inputs[i], cmap='tab10', vmin=0, vmax=9)
        axes[i][0].set_title(f"Input {i+1}")
        axes[i][0].axis('off')

        axes[i][1].imshow(outputs[i], cmap='tab10', vmin=0, vmax=9)
        axes[i][1].set_title(f"Output {i+1}")
        axes[i][1].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def main():
    print("="*60)
    print(f"Baseline NCA Training on Task {TASK_ID}")
    print("="*60)

    # Load task data
    print(f"\n[1/6] Loading task {TASK_ID}...")
    train_in, train_out, test_in, test_out = load_single_task(TASK_ID)

    # Generating data augmentations

    train_in = [
        np.rot90(arr, k=k).copy()
        for arr in train_in
        for k in range(4)
    ]

    train_out = [
        np.rot90(arr, k=k).copy()
        for arr in train_out
        for k in range(4)
    ]


    print(f"  - Training examples: {len(train_in)}")
    print(f"  - Test examples: {len(test_in)}")
    print(f"  - Training grid sizes: {[inp.shape for inp in train_in]}")
    print(f"  - Test grid size: {test_in[0].shape}")

    # Visualize the task
    print("\n[2/6] Visualizing task...")
    fig = visualize_arc_task(train_in, train_out, f"Task {TASK_ID} - Training")
    plt.savefig(OUTPUT_DIR / "training_examples.png", dpi=150, bbox_inches='tight')
    print(f"  - Saved: {OUTPUT_DIR / 'training_examples.png'}")

    fig = visualize_arc_task(test_in, test_out, f"Task {TASK_ID} - Test")
    plt.savefig(OUTPUT_DIR / "test_example.png", dpi=150, bbox_inches='tight')
    print(f"  - Saved: {OUTPUT_DIR / 'test_example.png'}")
    plt.close('all')

    # Convert to NCA space
    print("\n[3/6] Converting to NCA space...")
    max_colors = max(
        max(inp.max() for inp in train_in + test_in),
        max(out.max() for out in train_out + test_out)
    ) + 1
    print(f"  - Max colors: {max_colors}")

    mode = "rgb"
    genes = [i for i in range(GENESIZE)]

    # Convert numpy arrays to torch tensors (keep as integers for bitwise ops)
    nca_in = [
        aau.arc_to_nca_space(max_colors, torch.from_numpy(inp).long().to('cuda'), CHANNELS, GENESIZE,
                            mode=mode, gene_location=genes, is_invis=1)
        for inp in train_in
    ]
    nca_out = [
        aau.arc_to_nca_space(max_colors, torch.from_numpy(out).long().to('cuda'), CHANNELS, GENESIZE,
                            mode=mode, gene_location=genes, is_invis=1)
        for out in train_out
    ]

    test_nca_in = [
        aau.arc_to_nca_space(max_colors, torch.from_numpy(inp).long().to('cuda'), CHANNELS, GENESIZE,
                            mode=mode, gene_location=genes, is_invis=1)
        for inp in test_in
    ]
    test_nca_out = [
        aau.arc_to_nca_space(max_colors, torch.from_numpy(out).long().to('cuda'), CHANNELS, GENESIZE,
                            mode=mode, gene_location=genes, is_invis=1)
        for out in test_out
    ]

    target_size = [20, 20]

    print(f"  - Padding all grids to {target_size}")
    nca_in = [aau.pad_to_size(target_size, n) for n in nca_in]
    nca_out = [aau.pad_to_size(target_size, n) for n in nca_out]
    test_nca_in = [aau.pad_to_size(target_size, n) for n in test_nca_in]
    test_nca_out = [aau.pad_to_size(target_size, n) for n in test_nca_out]

    # Create NCA pools
    pool_x = [n.tile(1024, 1, 1, 1) for n in nca_in]
    pool_y = nca_out

    print(f"  - Pool shapes: {[p.shape for p in pool_x]}")
    print(f"  - Target shapes: {[p.shape for p in pool_y]}")
    print(f"  - Test input shape: {test_nca_in[0].shape}")
    print(f"  - Test output shape: {test_nca_out[0].shape}")

    # Initialize NCA
    print("\n[4/6] Initializing NCA...")
    nca = CA(CHANNELS, vft.GENE_HIDDEN_N + vft.GENE_PROP_HIDDEN_N)
    nca = nca.to(DEVICE)

    param_count = sum(p.numel() for p in nca.parameters())
    print(f"  - Model: CA (baseline NCA)")
    print(f"  - Parameters: {param_count:,}")
    print(f"  - Device: {DEVICE}")

    # Setup optimizer
    optim = torch.optim.AdamW(nca.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2000, gamma=0.3)

    # Training
    print("\n[5/6] Training...")
    print(f"  - Iterations: {TRAINING_ITERATIONS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")

    loss_log = []

    for iteration in range(TRAINING_ITERATIONS):
        nca.train()

        # Select training example (cycle through)
        idx_problem = iteration % len(pool_x)

        # Get batch
        with torch.no_grad():
            x_prob = pool_x[idx_problem]
            x, idxs = utils.get_batch(x_prob, nca_in[idx_problem].clone(),
                                     BATCH_SIZE, noise_level=0.2)
            y = pool_y[idx_problem].tile(BATCH_SIZE, 1, 1, 1)

            # Ensure tensors are on correct device
            x = x.to(DEVICE)
            y = y.to(DEVICE)

        # Forward pass with random number of steps
        n_steps = random.randrange(*STEPS_BETWEEN_ITERATIONS)

        for _ in range(n_steps):
            x = nca(x, 0.5)

        # Compute loss (MSE on first 4 channels - RGB + alpha)
        loss = ((y[:, :4, :, :]) - (x[:, :4, :, :])).pow(2).mean()

        # Backward pass with gradient normalization
        optim.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in nca.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        optim.step()
        scheduler.step()

        loss_log.append(loss.item())

        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"  Iter {iteration:4d} | Train Loss: {loss.item():.6f}")
            nca.eval()
            with torch.no_grad():
                test_x = test_nca_in[0].unsqueeze(0).clone().to(DEVICE)

                # Run NCA
                for _ in range(64):
                    test_x = nca(test_x, 1.0)

                # Convert to viewable image
                test_pred_img = aau.nca_to_rgb_image(test_x)
                test_true_img = aau.nca_to_rgb_image(test_nca_out[0])

                # Plot side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                ax1.imshow(test_pred_img)
                ax1.set_title("NCA Prediction")
                ax1.axis('off')

                ax2.imshow(test_true_img)
                ax2.set_title("Ground Truth")
                ax2.axis('off')

                plt.savefig(OUTPUT_DIR / f"test_prediction_{iteration}.png", dpi=150, bbox_inches='tight')
            nca.train()

    # After training completes
    print("\n[6/8] Generating test prediction...")
    nca.eval()
    with torch.no_grad():
        test_x = test_nca_in[0].unsqueeze(0).clone().to(DEVICE)

        # Run NCA
        for _ in range(64):
            test_x = nca(test_x, 1.0)

        # Convert to viewable image
        test_pred_img = aau.nca_to_rgb_image(test_x)
        test_true_img = aau.nca_to_rgb_image(test_nca_out[0])

        # Plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(test_pred_img)
        ax1.set_title("NCA Prediction")
        ax1.axis('off')

        ax2.imshow(test_true_img)
        ax2.set_title("Ground Truth")
        ax2.axis('off')

        plt.savefig(OUTPUT_DIR / "test_prediction_final.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / 'test_prediction_final.png'}")

    # Save model
    model_path = OUTPUT_DIR / "baseline_model.pth"
    torch.save(nca.state_dict(), model_path)
    print(f"\n[7/8] Model saved: {model_path}")

    # Plot training curves
    print("\n[8/8] Plotting results...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(loss_log, alpha=0.3, color='blue')
    ax.plot(np.convolve(loss_log, np.ones(50)/50, mode='valid'),
            color='blue', linewidth=2, label='Training Loss (smoothed)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150, bbox_inches='tight')
    print(f"  - Saved: {OUTPUT_DIR / 'training_curves.png'}")

    # Final evaluation
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Final training loss: {loss_log[-1]:.6f}")

    # Save logs
    results = {
        'task_id': TASK_ID,
        'iterations': TRAINING_ITERATIONS,
        'final_train_loss': loss_log[-1],
        'train_loss_history': loss_log,
    }

    with open(OUTPUT_DIR / "baseline_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
