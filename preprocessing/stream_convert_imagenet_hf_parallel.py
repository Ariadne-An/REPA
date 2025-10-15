"""
使用 HuggingFace streaming + 多进程并行下载

相比单线程版本，速度提升 4-8 倍

使用方法：
    python preprocessing/stream_convert_imagenet_hf_parallel.py \
        --max_images 200000 \
        --dest /runpod-volume/repa_sd15/data/images_512 \
        --resolution 512x512 \
        --transform center-crop \
        --batch 5000 \
        --workers 8
"""

import argparse
import tempfile
import subprocess
import sys
import json
import zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial

try:
    from datasets import load_dataset
except ImportError:
    print("❌ 错误: 未安装 datasets")
    print("   请运行: pip install datasets")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel Stream ImageNet from HuggingFace")
    parser.add_argument("--max_images", type=int, default=200000)
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--resolution", type=str, default="512x512")
    parser.add_argument("--transform", type=str, default="center-crop")
    parser.add_argument("--batch", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--work_dir", type=str, default="/workspace/hf_temp")
    return parser.parse_args()


def download_and_save(args_tuple):
    """
    Download and save a single image (worker function)

    Args:
        args_tuple: (example, count, buf_root)
    """
    example, count, buf_root = args_tuple

    try:
        img = example["image"]
        label = example.get("label", 0)

        # Create class directory
        class_dir = Path(buf_root) / f"class_{label:04d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        img_path = class_dir / f"img_{count:08d}.JPEG"

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(img_path, "JPEG")
        return True
    except Exception as e:
        print(f"⚠️  Error processing image {count}: {e}")
        return False


def flush_batch(buf_root, batch_dir, batch_id, resolution, transform):
    """Flush batch to ZIP"""
    if not any(buf_root.iterdir()):
        return None

    print(f"\n🖼️  Converting batch {batch_id} to {resolution}...")

    batch_output = batch_dir / f"batch_{batch_id:04d}"
    batch_output.mkdir(parents=True, exist_ok=True)

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
        print(f"⚠️  Warning: convert failed")
        return None

    # Clean buffer
    print("🗑️  Cleaning temp files...")
    for p in buf_root.rglob("*"):
        if p.is_file():
            p.unlink()

    for p in sorted(buf_root.glob("*"), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass

    batch_zip = batch_output / "dataset.zip"
    return batch_zip if batch_zip.exists() else None


def merge_batch_zips(batch_zips, output_path):
    """Merge all batch ZIPs"""
    print("\n" + "="*80)
    print("Merging batches...")
    print("="*80)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_labels = []
    image_count = 0

    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_STORED) as out_zf:
        for batch_zip in tqdm(batch_zips, desc="Merging"):
            if not batch_zip or not batch_zip.exists():
                continue

            with zipfile.ZipFile(batch_zip, 'r') as in_zf:
                if 'dataset.json' in in_zf.namelist():
                    dataset_json = json.loads(in_zf.read('dataset.json'))
                    batch_labels = dataset_json.get('labels', [])

                    for name in in_zf.namelist():
                        if name == 'dataset.json':
                            continue

                        data = in_zf.read(name)
                        ext = Path(name).suffix
                        new_name = f"{image_count:08d}{ext}"
                        out_zf.writestr(new_name, data)

                        for label_entry in batch_labels:
                            if label_entry[0] == name:
                                all_labels.append([new_name, label_entry[1]])
                                break

                        image_count += 1

        out_zf.writestr('dataset.json', json.dumps({'labels': all_labels}))

    print(f"\n✅ Merged {image_count} images to {output_path}")


def main():
    args = parse_args()

    print("="*80)
    print("HuggingFace ImageNet Parallel Streaming")
    print("="*80)
    print(f"Max images: {args.max_images}")
    print(f"Workers: {args.workers} (parallel)")
    print(f"Batch size: {args.batch}")
    print()

    # Setup directories
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    batch_dir = work_dir / "batches"
    batch_dir.mkdir(exist_ok=True)

    buf_root = work_dir / "temp_buffer"
    buf_root.mkdir(exist_ok=True)

    # Load dataset
    print("📥 Connecting to HuggingFace...")
    try:
        ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print("✅ Connected")
    print()

    # Collect examples in batches
    print("📊 Downloading images (parallel)...")

    batch_id = 0
    batch_zips = []
    examples_buffer = []
    count = 0

    with tqdm(total=args.max_images, desc="Processing") as pbar:
        for example in ds:
            if count >= args.max_images:
                break

            examples_buffer.append((example, count, buf_root))
            count += 1

            # Process batch when buffer is full
            if len(examples_buffer) >= args.batch:
                # Download in parallel
                with Pool(args.workers) as pool:
                    results = pool.map(download_and_save, examples_buffer)

                success_count = sum(results)
                pbar.update(success_count)

                # Convert to ZIP
                batch_zip = flush_batch(buf_root, batch_dir, batch_id,
                                       args.resolution, args.transform)
                if batch_zip:
                    batch_zips.append(batch_zip)

                batch_id += 1
                examples_buffer = []

        # Process remaining
        if examples_buffer:
            with Pool(args.workers) as pool:
                results = pool.map(download_and_save, examples_buffer)

            success_count = sum(results)
            pbar.update(success_count)

            batch_zip = flush_batch(buf_root, batch_dir, batch_id,
                                   args.resolution, args.transform)
            if batch_zip:
                batch_zips.append(batch_zip)

    # Merge all batches
    final_output = Path(args.dest) / "dataset.zip"
    merge_batch_zips(batch_zips, final_output)

    # Cleanup
    print("\n🗑️  Cleaning working directory...")
    import shutil
    try:
        shutil.rmtree(work_dir)
    except Exception as e:
        print(f"⚠️  Warning: {e}")

    print("\n✅ Complete!")
    print(f"Output: {final_output}")


if __name__ == "__main__":
    main()
