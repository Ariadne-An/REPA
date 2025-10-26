"""
Parallel ImageNet downloader with resume + batched conversion + final merge.

Workflow:
1. Stream ImageNet-1k from HuggingFace, save JPEGs into batch_xxxx/raw folders.
2. When a batch accumulates `batch_size` images, run dataset_tools.py convert to turn it
   into EDM2-compatible dataset.zip, then remove raw images to save space.
3. At the end, merge all batch-level dataset.zip files into a single dataset.zip.

Resume capability:
- On startup, scan existing batch directories under --work-dir
  * Already converted batches are preserved and included in final merge.
  * If a batch_xxxx/raw folder exists (incomplete), new images continue filling it.
  * Global image indexing (img_XXXXXXXX) resumes from the total count so far.

Usage example:
    python preprocessing/download_imagenet_parallel.py \
        --max-images 200000 \
        --dest data/images_512 \
        --resolution 512x512 \
        --transform center-crop-dhariwal \
        --batch-size 5000 \
        --workers 16 \
        --compress-level 0 \
        --work-dir /workspace/hf_parallel_buffer \
        --cleanup-work-dir
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile

from PIL import Image
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    print("❌ datasets 库未安装，请先运行 `pip install datasets`", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel HuggingFace ImageNet downloader")
    parser.add_argument('--max-images', type=int, default=200000, dest='max_images',
                        help='Total number of images to fetch (default: 200000)')
    parser.add_argument('--dest', type=str, required=True,
                        help='Destination directory; final dataset.zip will be placed here')
    parser.add_argument('--resolution', type=str, default='512x512',
                        help='Resolution passed to dataset_tools.py convert (default: 512x512)')
    parser.add_argument('--transform', type=str, default='center-crop-dhariwal',
                        help='Transform passed to dataset_tools.py convert (default: center-crop-dhariwal)')
    parser.add_argument('--batch-size', type=int, default=5000,
                        help='Number of images per flush/convert batch (default: 5000)')
    parser.add_argument('--workers', type=int, default=16,
                        help='Thread count for writing JPEG buffers (default: 16)')
    parser.add_argument('--compress-level', type=int, default=0,
                        help='Final ZIP compression level (0=stored, 1-9=DEFLATED)')
    parser.add_argument('--work-dir', type=str, default='/workspace/hf_parallel_buffer',
                        help='Working directory for temporary batches')
    parser.add_argument('--jpeg-quality', type=int, default=95,
                        help='Temporary JPEG quality before conversion (default: 95)')
    parser.add_argument('--hf-split', type=str, default='train',
                        help='HF split to stream (default: train)')
    parser.add_argument('--dataset-name', type=str, default='ILSVRC/imagenet-1k',
                        help='HF dataset identifier (default: ILSVRC/imagenet-1k)')
    parser.add_argument('--cleanup-work-dir', action='store_true',
                        help='Delete work-dir after successful merge (default: False)')
    return parser.parse_args()


def count_zip_images(zip_path: Path) -> int:
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if 'dataset.json' not in zf.namelist():
                return 0
            meta = json.loads(zf.read('dataset.json'))
            labels = meta.get('labels') or []
            return len(labels)
    except Exception:
        return 0


def gather_raw_entries(raw_dir: Path) -> List[List]:
    entries = []
    files = sorted([p for p in raw_dir.rglob('*') if p.suffix.lower() in {'.jpeg', '.jpg', '.png'} and p.is_file()])
    for file in files:
        rel = file.relative_to(raw_dir).as_posix()
        label = int(rel.split('/')[0]) if '/' in rel else int(file.parent.name)
        entries.append([rel, label])
    return entries


def prepare_resume_state(work_dir: Path, batch_size: int) -> Dict:
    work_dir.mkdir(parents=True, exist_ok=True)

    batch_dirs = sorted(work_dir.glob('batch_*'))
    batch_zips: List[Path] = []
    completed_ids: List[int] = []
    total_downloaded = 0

    resume_batch_id = None
    current_raw = None
    batch_labels: List[List] = []
    batch_count = 0

    for batch_dir in batch_dirs:
        try:
            batch_id = int(batch_dir.name.split('_')[-1])
        except ValueError:
            continue

        zip_path = batch_dir / 'dataset.zip'
        if zip_path.is_file():
            batch_zips.append(zip_path)
            count = count_zip_images(zip_path)
            total_downloaded += count
            completed_ids.append(batch_id)

        raw_dir = batch_dir / 'raw'
        if raw_dir.is_dir():
            existing_entries = gather_raw_entries(raw_dir)
            batch_labels = existing_entries
            batch_count = len(existing_entries)
            total_downloaded += batch_count
            resume_batch_id = batch_id
            current_raw = raw_dir

    if resume_batch_id is None:
        resume_batch_id = max(completed_ids, default=-1) + 1
        current_raw = work_dir / f'batch_{resume_batch_id:04d}' / 'raw'
        current_raw.mkdir(parents=True, exist_ok=True)
        batch_labels = []
        batch_count = 0
    else:
        current_raw.mkdir(parents=True, exist_ok=True)

    next_batch_id = max(completed_ids + [resume_batch_id], default=-1) + 1

    print(f"🔁 Resume summary: total_downloaded={total_downloaded}, "
          f"resume_batch_id={resume_batch_id:04d}, existing_completed={len(batch_zips)}")

    return dict(
        batch_zips=batch_zips,
        total_downloaded=total_downloaded,
        current_batch_id=resume_batch_id,
        next_batch_id=next_batch_id,
        current_raw=current_raw,
        batch_labels=batch_labels,
        batch_count=batch_count,
        work_dir=work_dir,
        batch_size=batch_size,
    )


def save_image_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)


def flush_batch(batch_id: int,
                raw_dir: Path,
                batch_labels: List[List],
                resolution: str,
                transform: str) -> Path:
    if not batch_labels:
        return None

    dataset_json_path = raw_dir / 'dataset.json'
    dataset_json_path.write_text(json.dumps({'labels': batch_labels}), encoding='utf-8')

    batch_root = raw_dir.parent
    batch_zip = batch_root / 'dataset.zip'

    cmd = [
        sys.executable,
        'preprocessing/dataset_tools.py',
        'convert',
        '--source', str(raw_dir),
        '--dest', str(batch_zip),
        '--resolution', resolution,
        '--transform', transform,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print('⚠️ convert 失败:')
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError('dataset_tools.py convert 执行失败')

    shutil.rmtree(raw_dir, ignore_errors=True)
    return batch_zip


def merge_batches(batch_zips: List[Path], final_zip: Path, compress_level: int):
    final_zip.parent.mkdir(parents=True, exist_ok=True)
    merged_labels = []
    next_idx = 0

    def new_name(idx: int) -> str:
        idx_str = f"{idx:08d}"
        return f"{idx_str[:5]}/img{idx_str}.png"

    compression = zipfile.ZIP_DEFLATED if compress_level > 0 else zipfile.ZIP_STORED
    compress_kwargs = {'compresslevel': compress_level} if compression == zipfile.ZIP_DEFLATED else {}

    with zipfile.ZipFile(final_zip, 'w', compression=compression, **compress_kwargs) as out_zf:
        for batch_zip in tqdm(batch_zips, desc='Merging batches'):
            if batch_zip is None or not batch_zip.exists():
                continue
            with zipfile.ZipFile(batch_zip, 'r') as in_zf:
                if 'dataset.json' not in in_zf.namelist():
                    continue
                dataset_meta = json.loads(in_zf.read('dataset.json'))
                labels = dataset_meta.get('labels', []) or []
                for old_name, label in labels:
                    data = in_zf.read(old_name)
                    fname = new_name(next_idx)
                    out_zf.writestr(fname, data)
                    merged_labels.append([fname, label])
                    next_idx += 1
        out_zf.writestr('dataset.json', json.dumps({'labels': merged_labels}))

    print(f"✅ 合并完成，共 {next_idx} 张，输出 {final_zip}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()

    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)
    final_zip = dest_root / 'dataset.zip'

    work_dir = Path(args.work_dir)
    state = prepare_resume_state(work_dir, args.batch_size)

    print('📥 连接 HuggingFace 数据集...')
    try:
        dataset = load_dataset(args.dataset_name, split=args.hf_split, streaming=True)
    except Exception as exc:  # pragma: no cover
        print(f"❌ 无法加载 {args.dataset_name}: {exc}")
        print('请确认已申请访问权限并运行 huggingface-cli login')
        sys.exit(1)

    executor = ThreadPoolExecutor(max_workers=args.workers)
    pending = []

    def flush_pending():
        for fut in pending:
            fut.result()
        pending.clear()

    batch_zips = list(state['batch_zips'])
    total = state['total_downloaded']
    current_batch_id = state['current_batch_id']
    next_batch_id = state['next_batch_id']
    current_raw = state['current_raw']
    batch_labels = list(state['batch_labels'])
    batch_count = state['batch_count']

    print(f"➡️ 从 total={total} 开始继续下载，当前 batch={current_batch_id:04d}, 已有 {batch_count} 张")

    try:
        with tqdm(total=args.max_images, initial=total, desc='Downloading') as pbar:
            for example in dataset:
                if total >= args.max_images:
                    break

                img = example['image']
                label = int(example.get('label', 0))
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                rel_path = Path(f"{label:04d}") / f"img_{total:08d}.jpeg"
                abs_path = current_raw / rel_path

                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=args.jpeg_quality, optimize=True)
                data = buf.getvalue()

                pending.append(executor.submit(save_image_bytes, abs_path, data))
                batch_labels.append([rel_path.as_posix(), label])

                total += 1
                batch_count += 1
                pbar.update(1)

                if batch_count >= args.batch_size:
                    flush_pending()
                    batch_zip = flush_batch(current_batch_id, current_raw, batch_labels,
                                            args.resolution, args.transform)
                    batch_zips.append(batch_zip)

                    current_batch_id = next_batch_id
                    next_batch_id += 1
                    current_raw = work_dir / f"batch_{current_batch_id:04d}" / 'raw'
                    current_raw.mkdir(parents=True, exist_ok=True)
                    batch_labels = []
                    batch_count = 0

        flush_pending()
        if batch_labels:
            batch_zip = flush_batch(current_batch_id, current_raw, batch_labels,
                                    args.resolution, args.transform)
            batch_zips.append(batch_zip)
            batch_labels = []

    finally:
        executor.shutdown(wait=True)

    if not batch_zips:
        print('❌ 没有 batch 转换成功，终止。')
        sys.exit(1)

    merge_batches(batch_zips, final_zip, args.compress_level)

    if args.cleanup_work_dir:
        print(f'🧹 清理工作目录: {work_dir}')
        shutil.rmtree(work_dir, ignore_errors=True)
        print('✅ 工作目录已清理')

    print('🎉 完成全部流程！')


if __name__ == '__main__':
    main()
