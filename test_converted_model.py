"""
Test the converted SD1.5 model.

Usage:
    python test_converted_model.py --model_path models/sd15_repa_step24k
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to converted model")
    parser.add_argument("--prompt", type=str, default="a photo of a cat",
                        help="Text prompt")
    parser.add_argument("--num_images", type=int, default=4,
                        help="Number of images to generate")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="CFG scale")
    parser.add_argument("--output", type=str, default="test_output.png",
                        help="Output image path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("Testing Converted SD1.5 Model")
    print("="*80)

    # Load pipeline
    print(f"\n✓ Loading model from {args.model_path}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")

    print(f"✓ Model loaded successfully!")

    # Generate
    print(f"\n✓ Generating {args.num_images} images...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Steps: {args.steps}")
    print(f"  CFG Scale: {args.guidance_scale}")

    generator = torch.Generator("cuda").manual_seed(args.seed)

    images = pipe(
        prompt=[args.prompt] * args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images

    # Save as grid
    print(f"\n✓ Saving to {args.output}...")
    grid_size = int(args.num_images ** 0.5)
    width, height = images[0].size

    grid = Image.new('RGB', (width * grid_size, height * grid_size))
    for i, img in enumerate(images):
        grid.paste(img, (i % grid_size * width, i // grid_size * height))

    grid.save(args.output)

    print(f"\n✅ Test complete! Check {args.output}")


if __name__ == "__main__":
    main()
