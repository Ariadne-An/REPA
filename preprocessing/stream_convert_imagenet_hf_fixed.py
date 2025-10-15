"""
使用 HuggingFace streaming 流式下载和处理 ImageNet（修复版）

修复：使用单独的批次目录，最后合并所有 ZIP

优势：
- 真正的流式处理，不需要下载完整数据集
- 临时空间由 batch_size 控制（5k 张 ≈ 2-3GB）
- 避免 dataset_tools.py 的覆盖问题

前置要求：
1. 申请 ImageNet-1k 访问权限：https://huggingface.co/datasets/ILSVRC/imagenet-1k
2. 登录 HuggingFace: huggingface-cli login

使用方法：
    python preprocessing/stream_convert_imagenet_hf_fixed.py \
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
import json
import zipfile
import io
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
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Working directory for temporary files (default: /workspace/hf_temp)"
    )
    return parser.parse_args()


def flush_batch(buf_root, batch_dir, batch_id, resolution, transform):
    """
    Flush current batch to a separate ZIP file

    Args:
        buf_root: Temporary buffer directory with raw images
        batch_dir: Directory to store batch ZIPs
        batch_id: Batch number
        resolution: Output resolution (e.g., "512x512")
        transform: Transform type (e.g., "center-crop")

    Returns:
        Path to batch ZIP file
    """
    # Check if buffer has any images
    if not any(buf_root.iterdir()):
        return None

    print(f"\n🖼️  Converting batch {batch_id} to {resolution}...")

    # Create batch output directory
    batch_output = batch_dir / f"batch_{batch_id:04d}"
    batch_output.mkdir(parents=True, exist_ok=True)

    # Call dataset_tools.py convert
    result = subprocess.run([
        sys.executable,
        "preprocessing/dataset_tools.py",
        "convert",
        "--source", str(buf_root),
        "--dest", str(batch_output / "dataset.zip"),
        "--resolution", resolution,
        "--transform", transform,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"⚠️  警告: convert 失败")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
        return None

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
                pass

    batch_zip = batch_output / "dataset.zip"
    if batch_zip.exists():
        print(f"✅ Batch {batch_id} 完成: {batch_zip}")
        return batch_zip
    else:
        return None


def merge_batch_zips(batch_zips, output_path):
    """
    Merge multiple batch ZIP files into one final dataset.zip

    Args:
        batch_zips: List of batch ZIP file paths
        output_path: Final output path (e.g., data/images_512/dataset.zip)
    """
    print("\n" + "="*80)
    print("合并所有批次...")
    print("="*80)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all labels from batch dataset.json files
    all_labels = []
    image_count = 0

    # Create final ZIP
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_STORED) as out_zf:
        for batch_zip in tqdm(batch_zips, desc="Merging batches"):
            if not batch_zip or not batch_zip.exists():
                continue

            # Read batch ZIP
            with zipfile.ZipFile(batch_zip, 'r') as in_zf:
                # Read dataset.json from this batch
                if 'dataset.json' in in_zf.namelist():
                    dataset_json = json.loads(in_zf.read('dataset.json'))
                    batch_labels = dataset_json.get('labels', [])

                    # Copy all image files
                    for name in in_zf.namelist():
                        if name == 'dataset.json':
                            continue

                        # Read and write image
                        data = in_zf.read(name)

                        # Generate new filename with global index
                        ext = Path(name).suffix
                        new_name = f"{image_count:08d}{ext}"

                        out_zf.writestr(new_name, data)

                        # Update label with new filename
                        # Find corresponding label in batch_labels
                        old_name = name
                        for label_entry in batch_labels:
                            if label_entry[0] == old_name:
                                all_labels.append([new_name, label_entry[1]])
                                break

                        image_count += 1

        # Write final dataset.json
        final_dataset_json = {
            'labels': all_labels
        }
        out_zf.writestr('dataset.json', json.dumps(final_dataset_json))

    print(f"\n✅ 合并完成:")
    print(f"   总图片数: {image_count}")
    print(f"   输出: {output_path}")
    print(f"   大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    args = parse_args()

    print("="*80)
    print("HuggingFace ImageNet Streaming Converter (Fixed)")
    print("="*80)
    print(f"Max images: {args.max_images}")
    print(f"Destination: {args.dest}")
    print(f"Resolution: {args.resolution}")
    print(f"Transform: {args.transform}")
    print(f"Batch size: {args.batch}")
    print()

    # Setup working directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = Path("/workspace/hf_temp")

    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 工作目录: {work_dir}")

    # Create batch directory
    batch_dir = work_dir / "batches"
    batch_dir.mkdir(exist_ok=True)

    # Create temporary buffer directory
    buf_root = work_dir / "temp_buffer"
    buf_root.mkdir(exist_ok=True)
    print(f"📁 临时缓冲: {buf_root}")
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
    batch_id = 0  # Current batch ID
    batch_zips = []  # List of batch ZIP files

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
                    batch_zip = flush_batch(
                        buf_root,
                        batch_dir,
                        batch_id,
                        args.resolution,
                        args.transform
                    )
                    if batch_zip:
                        batch_zips.append(batch_zip)

                    batch_id += 1
                    batch_count = 0

            except Exception as e:
                print(f"\n⚠️  警告: 处理第 {count} 张图片时出错: {e}")
                continue

    # Flush remaining images
    if batch_count > 0:
        print(f"\n🖼️  处理最后一批 ({batch_count} 张)...")
        batch_zip = flush_batch(
            buf_root,
            batch_dir,
            batch_id,
            args.resolution,
            args.transform
        )
        if batch_zip:
            batch_zips.append(batch_zip)

    # Merge all batch ZIPs into final dataset.zip
    final_output = Path(args.dest) / "dataset.zip"
    merge_batch_zips(batch_zips, final_output)

    # Clean up working directory
    print("\n🗑️  清理工作目录...")
    import shutil
    try:
        shutil.rmtree(work_dir)
        print(f"✅ 已删除: {work_dir}")
    except Exception as e:
        print(f"⚠️  警告: 无法删除工作目录: {e}")

    print()
    print("="*80)
    print("✅ 完成!")
    print("="*80)
    print(f"处理了 {count} 张图片")
    print(f"输出: {final_output}")
    print()


if __name__ == "__main__":
    main()
