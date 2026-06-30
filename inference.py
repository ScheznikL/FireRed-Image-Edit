"""FireRed-Image-Edit Inference Demo."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from io import BytesIO
import requests
from urllib.parse import urlparse

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FireRed-Image-Edit inference script"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="FireRedTeam/FireRed-Image-Edit-1.0",
        help="Path to the model or HuggingFace model ID",
    )
    # parser.add_argument(
    #     "--input_image",
    #     type=Path,
    #     nargs="+",
    #     default=[Path("./examples/edit_example.png")],
    #     help="Path(s) to the input image(s). Supports 1-N images. "
    #          "When more than 3 images are given the agent will "
    #          "automatically crop and stitch them into 2-3 composites.",
    # )
    parser.add_argument(
        "--input_image",
        type=str,
        nargs="+",
        default=["./examples/edit_example.png"],
        help="Path(s) to the input image(s). Supports 1-N images. "
             "When more than 3 images are given the agent will "
             "automatically crop and stitch them into 2-3 composites.",
    )
    parser.add_argument(
        "--output_image",
        type=Path,
        default=Path("output_edit.png"),
        help="Path to save the output image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="在书本封面Python的下方，添加一行英文文字2nd Edition",
        help="Editing prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="True CFG scale",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--recaption",
        action="store_true",
        default=False,
        help="Enable agent-based recaption: expand the editing prompt to "
             "~512 words/characters via Gemini for richer context. "
             "Requires GEMINI_API_KEY environment variable.",
    )
    return parser.parse_args()


def load_pipeline(model_path: str) -> QwenImageEditPlusPipeline:
    """Load FireRed image edit pipeline."""
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        #device_map = "balanced"
    )
    
    #pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
   # pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)
    return pipe

def prepare_qwen_image(img, max_side=1024):
    w, h = img.size
    
    # scale so longest side ≈ max_side
    scale = max_side / max(w, h)
    w = int(w * scale)
    h = int(h * scale)
    
    # Qwen-VL requirement: multiple of 112
    w = (w // 112) * 112
    h = (h // 112) * 112
    
    return img.resize((w, h))

def load_image(p):
    parsed = urlparse(p)
    if parsed.scheme in ("http", "https"):
        r = requests.get(p, timeout=30)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    return Image.open(p).convert("RGB")



def main() -> None:
    """Main entry point."""
    args = parse_args()

    pipeline = load_pipeline(args.model_path)
    print("Pipeline loaded.")

    # ── Load all input images ──
    #images = [Image.open(p).convert("RGB") for p in args.input_image] Add a check to then load from network
    images = [load_image(p) for p in args.input_image]
    
    # 1. UNCOMMENTED: This correctly resizes and enforces the 112-multiple rule.
    images = [prepare_qwen_image(img) for img in images]

    for i, im in enumerate(images):
        print("Prepared size:", im.size)    
    
    # 2. DELETED the manual MAX_SIDE block here because prepare_qwen_image 
    # already scales the longest side to 1024 safely.

    prompt = args.prompt
    #prompt = f"<|vision_start|><|image_pad|><|vision_end|>{args.prompt}"
    print(f"Loaded {len(images)} image(s).")

    # ── Agent: recaption only (since we only have 1 image, stitch is False) ──
    need_stitch = len(images) > 3
    need_recaption = args.recaption

    if need_stitch or need_recaption:
        from agent import AgentPipeline

        agent = AgentPipeline(verbose=True)
        agent_result = agent.run(
            images,
            prompt,
            enable_recaption=need_recaption or need_stitch,
        )
        images = agent_result.images
        prompt = agent_result.prompt
        print(f"Agent produced {len(images)} image(s).")
        print(f"Rewritten prompt: {prompt[:200]}{'…' if len(prompt) > 200 else ''}")

    inputs = {
        "image": images[0],
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
    }

    with torch.inference_mode():
        result = pipeline(**inputs)

    output_image = result.images[0]
    output_image.save(args.output_image)

    print("Image saved at:", args.output_image.resolve())
    
# def main() -> None:
#     """Main entry point."""
#     args = parse_args()

#     pipeline = load_pipeline(args.model_path)
   
    
#     print("Pipeline loaded.")

#     # ── Load all input images ──
#     images = [Image.open(p).convert("RGB") for p in args.input_image]
#     #images = [prepare_qwen_image(img) for img in images]

#     for i, im in enumerate(images):
#         print("Prepared size:", im.size)    
    
#     # MAX_SIDE = 1024
#     # images = [img.resize(
#     #     (int(img.width * MAX_SIDE / max(img.size)),
#     #     int(img.height * MAX_SIDE / max(img.size))))
#     #     if max(img.size) > MAX_SIDE else img
#     #     for img in images
#     # ]
#     prompt = args.prompt
#     print(f"Loaded {len(images)} image(s).")

#     # ── Agent: stitch + recaption when needed ──
#     need_stitch = len(images) > 3
#     need_recaption = args.recaption

#     if need_stitch or need_recaption:
#         from agent import AgentPipeline

#         agent = AgentPipeline(verbose=True)
#         agent_result = agent.run(
#             images,
#             prompt,
#             enable_recaption=need_recaption or need_stitch,
#         )
#         images = agent_result.images
#         prompt = agent_result.prompt
#         print(f"Agent produced {len(images)} image(s).")
#         print(f"Rewritten prompt: {prompt[:200]}{'…' if len(prompt) > 200 else ''}")

#     inputs = {
#         "image": images,
#         "prompt": prompt,
#         "generator": torch.Generator(device="cuda").manual_seed(args.seed),
#         "true_cfg_scale": args.true_cfg_scale,
#         "negative_prompt": " ",
#         "num_inference_steps": args.num_inference_steps,
#         "num_images_per_prompt": 1,
#     }

#     with torch.inference_mode():
#         result = pipeline(**inputs)

#     output_image = result.images[0]
#     output_image.save(args.output_image)

#     print("Image saved at:", args.output_image.resolve())


if __name__ == "__main__":
    main()