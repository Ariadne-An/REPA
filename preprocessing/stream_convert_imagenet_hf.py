"""
使用 HuggingFace streaming 流式下载和处理 ImageNet

优势：
- 真正的流式处理，不需要下载完整数据集
- 临时空间由 batch_size 控制（5k 张 ≈ 2-3GB）
- 直接调用现有的 dataset_tools.py convert

前置要求：
1. 申请 ImageNet-1k 访问权限：https://huggingface.co/datasets/ILSVRC/imagenet-1k
2. 登录 HuggingFace: huggingface-cli login

使用方法：
    python preprocessing/stream_convert_imagenet_hf.py \
        --max_images 200000 \
        --dest /runpod-volume/repa_sd15/data/images_512 \
        --resolution 512x512 \
        --transform center-crop \
        --batch 5000
"""

import argparse
import tempfile
import subprocess
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 检查依赖
try:
    from datasets import load_dataset
except ImportError:
    print("❌ 错误: 未安装 datasets")
    print("   请运行: pip install datasets")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Stream ImageNet from HuggingFace")
    parser.add_argument(
        "--max_images",
        type=int,
        default=200000,
        help="Maximum number of images to process (default: 200000)"
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Output directory (e.g., data/images_512)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="512x512",
        help="Output resolution (default: 512x512)"
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="center-crop",
        help="Transform type (default: center-crop)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=5000,
        help="Batch size for flushing (default: 5000)"
    )
    return parser.parse_args()


def flush_batch(buf_root, dest, resolution, transform):
    """
    Flush current batch to dataset.zip using dataset_tools.py convert

    Args:
        buf_root: Temporary buffer directory
        dest: Destination directory
        resolution: Output resolution (e.g., "512x512")
        transform: Transform type (e.g., "center-crop")
    """
    # Check if buffer has any images
    if not any(buf_root.iterdir()):
        return

    print(f"\n🖼️  Converting batch to {resolution}...")

    # Call dataset_tools.py convert
    result = subprocess.run([
        sys.executable,  # Use current Python interpreter
        "preprocessing/dataset_tools.py",
        "convert",
        "--source", str(buf_root),
        "--dest", str(dest),
        "--resolution", resolution,
        "--transform", transform,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"⚠️  警告: convert 失败")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
        # Continue anyway

    # Clean buffer directory
    print("🗑️  清理临时文件...")
    for p in buf_root.rglob("*"):
        if p.is_file():
            p.unlink()

    # Remove empty directories
    for p in sorted(buf_root.glob("*"), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass  # Directory not empty, skip


def main():
    args = parse_args()

    print("="*80)
    print("HuggingFace ImageNet Streaming Converter")
    print("="*80)
    print(f"Max images: {args.max_images}")
    print(f"Destination: {args.dest}")
    print(f"Resolution: {args.resolution}")
    print(f"Transform: {args.transform}")
    print(f"Batch size: {args.batch}")
    print()

    # Create destination directory
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Create temporary buffer directory
    buf_root = Path(tempfile.mkdtemp(prefix="hf_imagenet_"))
    print(f"📁 临时目录: {buf_root}")
    print()

    # Load dataset in streaming mode
    print("📥 连接 HuggingFace ImageNet-1k (streaming mode)...")
    try:
        ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    except Exception as e:
        print(f"❌ 错误: 无法加载数据集")
        print(f"   {e}")
        print()
        print("请确保:")
        print("  1. 已申请 ImageNet-1k 访问权限:")
        print("     https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("  2. 已登录 HuggingFace:")
        print("     huggingface-cli login")
        sys.exit(1)

    print("✅ 连接成功，开始处理...")
    print()

    count = 0  # Total images processed
    batch_count = 0  # Images in current batch

    # Process streaming dataset
    with tqdm(total=args.max_images, desc="Processing images") as pbar:
        for example in ds:
            if count >= args.max_images:
                break

            try:
                # Get image and label
                img = example["image"]
                label = example.get("label", 0)

                # Create class directory
                class_dir = buf_root / f"class_{label:04d}"
                class_dir.mkdir(parents=True, exist_ok=True)

                # Save image
                img_path = class_dir / f"img_{count:08d}.JPEG"

                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                img.save(img_path, "JPEG")

                count += 1
                batch_count += 1
                pbar.update(1)

                # Flush batch if reached batch size
                if batch_count >= args.batch:
                    flush_batch(buf_root, dest, args.resolution, args.transform)
                    batch_count = 0

            except Exception as e:
                print(f"\n⚠️  警告: 处理第 {count} 张图片时出错: {e}")
                continue

    # Flush remaining images
    if batch_count > 0:
        print(f"\n🖼️  处理最后一批 ({batch_count} 张)...")
        flush_batch(buf_root, dest, args.resolution, args.transform)

    # Clean up temporary directory
    print("\n🗑️  清理临时目录...")
    try:
        buf_root.rmdir()
    except OSError:
        print(f"⚠️  警告: 无法删除临时目录 {buf_root}")

    print()
    print("="*80)
    print("✅ 完成!")
    print("="*80)
    print(f"处理了 {count} 张图片")
    print(f"输出: {dest}/dataset.zip")
    print()


if __name__ == "__main__":
    main()
