"""
Example 6: Training a Custom Face Swap Model

Demonstrates the end-to-end custom model training workflow
including dataset setup, training, and ONNX export.

Usage:
    python examples/06_train_model.py \
        --dataset ./data/faces \
        --epochs 50 \
        --resolution 256
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Train Custom Face Swap Model")
    parser.add_argument("--dataset", required=True, help="Training images directory")
    parser.add_argument("--output", default="./training_output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--export", default="./models/custom_swap.onnx")
    args = parser.parse_args()

    print("=" * 50)
    print("  Face Swap Model Training")
    print("=" * 50)
    print()

    # ── Step 1: Configure training ──
    from face_swap.training import FaceSwapTrainer, TrainingConfig

    if FaceSwapTrainer is None:
        print("Error: PyTorch is required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    config = TrainingConfig(
        dataset_dir=args.dataset,
        output_dir=args.output,
        model_arch="simswap",
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=1e-4,
        mixed_precision=True,
        device=args.device,
        tensorboard=True,
    )

    print("Training Configuration:")
    print(f"  Dataset:     {config.dataset_dir}")
    print(f"  Architecture: {config.model_arch}")
    print(f"  Resolution:  {config.resolution}×{config.resolution}")
    print(f"  Batch size:  {config.batch_size}")
    print(f"  Epochs:      {config.num_epochs}")
    print(f"  Device:      {config.device}")
    print(f"  Precision:   {'FP16 (mixed)' if config.mixed_precision else 'FP32'}")
    print()

    # ── Step 2: Prepare dataset ──
    print("Dataset Structure Expected:")
    print(f"  {config.dataset_dir}/")
    print(f"  ├── identity_001/")
    print(f"  │   ├── img_001.jpg")
    print(f"  │   ├── img_002.jpg")
    print(f"  │   └── ...")
    print(f"  ├── identity_002/")
    print(f"  │   └── ...")
    print(f"  └── ...")
    print()

    # ── Step 3: Train ──
    print("Starting training...")
    print("(Monitor with: tensorboard --logdir ./training_output/tensorboard)")
    print()

    trainer = FaceSwapTrainer(config)

    try:
        state = trainer.train()
        print()
        print(f"✅ Training complete!")
        print(f"  Final loss:  {state.loss_history[-1].get('total', 'N/A')}")
        print(f"  Best loss:   {state.best_loss:.4f}")
        print(f"  Total steps: {state.global_step}")
        print()

        # ── Step 4: Export to ONNX ──
        print(f"Exporting to ONNX: {args.export}")
        trainer.export_onnx(args.export)
        print(f"✅ Model exported: {args.export}")
        print()
        print("Next steps:")
        print("  1. Optimise with TensorRT:")
        print(f"     python -m face_swap.optimization.export_cli \\")
        print(f"         --onnx {args.export} --engine model_fp16.engine --precision fp16")
        print()
        print("  2. Use in the pipeline:")
        print(f'     config = FaceSwapConfig(swap_model_path="{args.export}")')

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted. Checkpoints saved in:", config.output_dir)


if __name__ == "__main__":
    main()
