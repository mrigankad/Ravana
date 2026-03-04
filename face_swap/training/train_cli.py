"""
CLI for model training.

Usage:
    python -m face_swap.training.train_cli \
        --dataset ./data/faces \
        --output ./training_output \
        --epochs 100 \
        --batch-size 8 \
        --resolution 256 \
        --precision fp16

    python -m face_swap.training.train_cli \
        --resume ./training_output/best.pth \
        --export ./models/custom_swap.onnx
"""

import argparse
import logging
import sys

from .trainer import FaceSwapTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom face swap model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--dataset", required=True, help="Training images directory")
    parser.add_argument(
        "--output", default="./training_output", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--arch", choices=["simswap", "aei_net"], default="simswap")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--export", help="Export to ONNX after training")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--no-tensorboard", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = TrainingConfig(
        dataset_dir=args.dataset,
        output_dir=args.output,
        model_arch=args.arch,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        mixed_precision=(args.precision == "fp16"),
        resume_from=args.resume,
        device=args.device,
        num_workers=args.workers,
        tensorboard=not args.no_tensorboard,
    )

    trainer = FaceSwapTrainer(config)

    try:
        state = trainer.train()
        print(f"\n✅ Training complete. Best loss: {state.best_loss:.4f}")

        if args.export:
            trainer.export_onnx(args.export)
            print(f"✅ Model exported: {args.export}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
