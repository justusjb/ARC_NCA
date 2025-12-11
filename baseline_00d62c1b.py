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
import cv2
import shutil

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
TRAINING_ITERATIONS = 5000
LEARNING_RATE = 5e-3 # lowered from 1e-3
STEPS_BETWEEN_ITERATIONS = (20, 31)  # Random range, originally 32,64, now always 10.
# Curiously, this originally always made 64 steps at eval but at most 63 when training
EVAL_STEPS = STEPS_BETWEEN_ITERATIONS[1] - 1
MODE = "onehot"

# Paths
DATA_ROOT = Path("ArcData/data")
TRAINING_PATH = DATA_ROOT / "training"
OUTPUT_DIR = Path("results") / TASK_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_PHOTOS = OUTPUT_DIR / "photos"
OUTPUT_DIR_PHOTOS.mkdir(parents=True, exist_ok=True)
path_video = Path("results") / TASK_ID / "video"
path_video.mkdir(parents=True, exist_ok=True)

ARC_COLOR_MAP_NP = np.array([
    [0, 0, 0],       # 0: black
    [0, 116, 217],   # 1: blue
    [255, 65, 54],   # 2: red
    [46, 204, 64],   # 3: green
    [255, 220, 0],   # 4: yellow
    [170, 170, 170], # 5: gray
    [240, 18, 190],  # 6: magenta
    [255, 133, 27],  # 7: orange
    [127, 219, 255], # 8: light blue
    [135, 60, 0]     # 9: brown
], dtype=np.float32) / 255.0

ARC_COLOR_MAP_TORCH = torch.from_numpy(ARC_COLOR_MAP_NP)


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


def arc_to_rgb_display(arc_grid):
    """Convert raw ARC grid (0-9 integers) to RGB using ARC color palette"""
    return ARC_COLOR_MAP_NP[arc_grid]


def visualize_results(nca, train_in, train_out, test_in, test_out,
                      nca_train_in, nca_train_out, nca_test_in, nca_test_out, eval_steps=EVAL_STEPS, mode="rgb"):
    """Visualize training and test predictions vs ground truth"""

    nca.eval()
    with torch.no_grad():
        # Generate predictions for all training examples
        train_preds = []
        for x in nca_train_in:
            x_pred = x.unsqueeze(0).clone().to(DEVICE)
            for _ in range(eval_steps):
                x_pred = nca(x_pred, 1.0)
            train_preds.append(x_pred.squeeze(0))

        # Generate test prediction
        test_x = nca_test_in[0].unsqueeze(0).clone().to(DEVICE)
        for _ in range(eval_steps):
            test_x = nca(test_x, 1.0)
        test_pred = test_x.squeeze(0)

    # Convert to images
    train_pred_imgs = [aau.nca_to_rgb_image(p, mode=mode) for p in train_preds]

    # Plot everything
    n_train = len(train_in)
    fig, axes = plt.subplots(n_train, 3, figsize=(12, 4 * n_train))

    # Training examples
    for i in range(n_train):
        # Input (raw ARC)
        axes[i, 0].imshow(arc_to_rgb_display(train_in[i]))
        axes[i, 0].set_title(f"Train {i + 1} Input")
        axes[i, 0].axis('off')

        # Prediction
        axes[i, 1].imshow(np.clip(train_pred_imgs[i], 0, 1))
        axes[i, 1].set_title(f"Train {i + 1} Prediction")
        axes[i, 1].axis('off')

        # Ground truth (raw ARC)
        axes[i, 2].imshow(arc_to_rgb_display(train_out[i]))
        axes[i, 2].set_title(f"Train {i + 1} Truth")
        axes[i, 2].axis('off')

    plt.tight_layout()
    return fig


def write_frame(x, path, frame_number, height, width, chn, mode="rgb"):
    if mode == "onehot":
        # x shape: [B, C, H, W]
        # Argmax over one-hot channels (0-9)
        color_indices = torch.argmax(x[0, :10, :, :], dim=0).cpu()  # [H, W]
        rgb = ARC_COLOR_MAP_TORCH[color_indices]  # [H, W, 3]

        # Apply alpha mask
        alpha = x[0, 10, :, :].unsqueeze(-1).cpu()  # [H, W, 1]
        rgb = rgb * alpha

        image_np = rgb.permute(1, 0, 2).numpy().clip(0, 1)
    elif mode == "rgb":
        image_np = x.clone().detach().cpu().permute(0,3,2,1).numpy().clip(0,1)[0,:,:,:3]
    else:
        raise NotImplementedError("Only rgb and onehot are supported")
    plt.imsave(f"{path}/frame_{frame_number}.png", image_np)


def make_video(path, total_frames, height, width, vid_num = "r"):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = Path(path) / f'{vid_num}.mp4'
    out = cv2.VideoWriter(output_path, fourcc, 15.0, (height, width))
    for frame_number in range(total_frames):
       frame_path = Path(path) / f"frame_{frame_number}.png"
       frame = cv2.imread(frame_path)
       #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       frame = cv2.flip(frame,1)
       frame = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
       frame = cv2.resize(frame, (height, width), interpolation=cv2.INTER_NEAREST)

       # Add text after upscaling - top right corner
       text = f'step {frame_number+1}'
       text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
       text_x = width - text_size[0] - 3  # 3 pixels from right edge
       text_y = text_size[1] + 3  # 3 pixels from top

       cv2.putText(frame,
                   text,
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.3,  # Smaller font scale
                   (255, 255, 255),
                   1,  # Thinner
                   cv2.LINE_AA)

       out.write(frame)
    out.release()


def make_circle_masks(n, h, w, device):
    """
    Create n random circular damage masks of size (h, w).

    Args:
        n: Number of masks to create
        h: Height of each mask
        w: Width of each mask
        device: torch device (e.g., 'cuda' or 'cpu')

    Returns:
        Tensor of shape [n, h, w] where:
        - 0.0 = inside circle (damaged region, will be zeroed out)
        - 1.0 = outside circle (preserved region)
    """
    # Create coordinate grid from -1 to 1
    x = torch.linspace(-1.0, 1.0, w, device=device)[None, None, :]  # [1, 1, w]
    y = torch.linspace(-1.0, 1.0, h, device=device)[None, :, None]  # [1, h, 1]

    # Random circle centers (offset from grid center)
    center = torch.rand(2, n, 1, 1, device=device) * 1.0 - 0.5  # [-0.5, 0.5]

    # Random radii
    r = torch.rand(n, 1, 1, device=device) * 0.15 + 0.05  # [0.05, 0.2] = .5px - 2px damage circle radius

    # Compute normalized distance from center
    x_dist = (x - center[0]) / r  # Broadcasting: [n, 1, w]
    y_dist = (y - center[1]) / r  # Broadcasting: [n, h, 1]

    # Circle mask: 1 inside circle, 0 outside
    circle = (x_dist * x_dist + y_dist * y_dist < 1.0).float()

    # Damage mask: 0 inside (damaged), 1 outside (preserved)
    damage_mask = 1.0 - circle

    return damage_mask  # [n, h, w]


def main():
    print("="*60)
    print(f"Baseline NCA Training on Task {TASK_ID}")
    print("="*60)

    # Load task data
    print(f"\n[1/6] Loading task {TASK_ID}...")
    train_in, train_out, test_in, test_out = load_single_task(TASK_ID)

    # Generating data augmentations

    use_augmentation = True
    use_flips = False

    if use_augmentation:
        train_in = [
            np.rot90(flipped_arr, k=k).copy()
            for arr in train_in
            for flipped_arr in ([arr, np.flip(arr, axis=1)] if use_flips else [arr])
            for k in range(4)
        ]

        train_out = [
            np.rot90(flipped_arr, k=k).copy()
            for arr in train_out
            for flipped_arr in ([arr, np.flip(arr, axis=1)] if use_flips else [arr])
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

    mode = MODE
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=TRAINING_ITERATIONS, eta_min=1e-5)

    ema_nca = torch.optim.swa_utils.AveragedModel(nca, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

    # Training
    print("\n[5/6] Training...")
    print(f"  - Iterations: {TRAINING_ITERATIONS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")

    loss_log = []

    for iteration in range(TRAINING_ITERATIONS):
        nca.train()

        # Select training example (choose random problem)
        idx_problem = random.randrange(len(pool_x))

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

        total_loss=0

        damage_step = -1
        if random.random() < 0.05:  # Only ~5% of all training iterations
            damage_step = random.randint(10, max(11, n_steps - 5))

        for i in range(n_steps):

            if i == damage_step:
                # Damage only 1-2 samples, not all
                n_to_damage = random.randint(1, 2)
                h, w = x.shape[2:]
                damage_masks = make_circle_masks(n_to_damage, h, w, DEVICE).unsqueeze(1)
                x[-n_to_damage:] *= damage_masks

            x = nca(x, 0.75)

            if i in [n_steps-1]:
                if MODE == "rgb":
                    step_loss = ((y[:, :4, :, :]) - (x[:, :4, :, :])).pow(2).mean()
                elif MODE == "onehot":
                    step_loss = ((y[:, :11, :, :]) - (x[:, :11, :, :])).pow(2).mean()
                else:
                    raise NotImplementedError("Only rgb and onehot are supported")
                total_loss = total_loss + step_loss

        # Compute loss (MSE on first 4 channels - RGB + alpha)
        # total_loss+= ((y[:, :4, :, :]) - (x[:, :4, :, :])).pow(2).mean()

        loss = total_loss
        # Backward pass with gradient normalization
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nca.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()

        ema_nca.update_parameters(nca)

        loss_log.append(loss.item())

        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"  Iter {iteration:4d} | Train Loss: {loss.item():.6f}")
            ema_nca.eval()
            with torch.no_grad():
                test_x = test_nca_in[0].unsqueeze(0).clone().to(DEVICE)

                # Run NCA
                for i in range(EVAL_STEPS+100):
                    test_x = ema_nca.module(test_x, 0.5)
                    if i == EVAL_STEPS-1:
                        test_pred_img1 = aau.nca_to_rgb_image(test_x, mode=MODE)
                    if i== EVAL_STEPS+19:
                        test_pred_img2 = aau.nca_to_rgb_image(test_x, mode=MODE)

                # Convert to viewable image
                test_pred_img3 = aau.nca_to_rgb_image(test_x, mode=MODE)
                test_true_img = aau.nca_to_rgb_image(test_nca_out[0], mode=MODE)

                # Plot side by side
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 8))
                ax1.imshow(np.clip(test_pred_img1, 0, 1))
                ax1.set_title("NCA Prediction")
                ax1.axis('off')

                ax2.imshow(np.clip(test_pred_img2, 0, 1))
                ax2.set_title("NCA Prediction 20 steps later")
                ax2.axis('off')

                ax3.imshow(np.clip(test_pred_img3, 0, 1))
                ax3.set_title("NCA Prediction 100 steps later")
                ax3.axis('off')

                ax4.imshow(np.clip(test_true_img, 0, 1))
                ax4.set_title("Ground Truth")
                ax4.axis('off')

                plt.savefig(OUTPUT_DIR_PHOTOS / f"test_prediction_{iteration}.png", dpi=150, bbox_inches='tight')
                plt.close()
            ema_nca.train()

    # After training completes
    print("\n[6/8] Generating test prediction...")

    ema_nca.eval()
    nca.eval()

    print("NCA params:", nca.w1.weight[0, 0, 0])

    # EMA model parameters (smoothed average)
    print("EMA NCA params:", ema_nca.module.w1.weight[0, 0, 0])


    with torch.no_grad():
        fig = visualize_results(ema_nca.module, train_in, train_out, test_in, test_out,
                                nca_in, nca_out, test_nca_in, test_nca_out, mode=MODE)
        plt.savefig(OUTPUT_DIR / "all_predictions.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {OUTPUT_DIR / 'all_predictions.png'}")
        plt.close('all')


        test_x = test_nca_in[0].unsqueeze(0).clone().to(DEVICE)

        # Run NCA
        for i in range(EVAL_STEPS+1000):
            test_x = ema_nca.module(test_x, 0.5)
            x = test_x.detach()
            write_frame(x, path_video, i, 10 * x.shape[3], 10 * x.shape[2], CHANNELS, mode=MODE)

        make_video(path_video, EVAL_STEPS+1000, 10 * x.shape[3], 10 * x.shape[2],
                   type(nca).__name__ + "problem_" + str(TASK_ID) + "padded")

        # Convert to viewable image
        test_pred_img = aau.nca_to_rgb_image(test_x, mode=MODE)
        test_true_img = aau.nca_to_rgb_image(test_nca_out[0], mode=MODE)

        # Plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(np.clip(test_pred_img, 0, 1))
        ax1.set_title("NCA Prediction")
        ax1.axis('off')

        ax2.imshow(np.clip(test_true_img,0,1))
        ax2.set_title("Ground Truth")
        ax2.axis('off')

        plt.savefig(OUTPUT_DIR_PHOTOS / "test_prediction_final.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR_PHOTOS / 'test_prediction_final.png'}")

    shutil.make_archive(
        str(OUTPUT_DIR_PHOTOS),  # Output path without extension
        'zip',  # Archive format
        str(OUTPUT_DIR_PHOTOS)  # Directory to zip
    )

    # Save model
    model_path = OUTPUT_DIR / "baseline_model.pth"
    torch.save(ema_nca.state_dict(), model_path)
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
