import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import torch
import argparse
from pipelines.pipeline_regional_guidance import RegionallyT2IAdapterPipeline
from boxnet_models import build_model, add_boxnet_args
from utils.utils import save_colored_mask, save_bbox_img, save_img_with_box
import numpy as np

OUTPUT_DIR='./results'
BOXNET_MODEL_PATH='./ckpt/boxnet.pt'
STABLE_MODEL_PATH='/data/zsz/ssh/models/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', default='test_prompts.json', type=str,
                        help="Prompt file to generate images.")
    parser.add_argument('--stable_model_path', required=False, default=STABLE_MODEL_PATH, 
                        type=str, help="Original stable diffusion model path.")
    parser.add_argument('--boxnet_model_path', required=False, default=BOXNET_MODEL_PATH,
                        type=str, help="BoxNet model path.")
    parser.add_argument('--output_dir', required=False, type=str, default=OUTPUT_DIR,
                        help="Output dir for results.")
    parser.add_argument('--seed', default=0, type=int,
                        help="Random seed.")
    parser.add_argument('--max_guidance_rate', default=0.5, type=float,
                        help="The rate using boxnet prediction.")
    parser = add_boxnet_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda")
    with open(args.prompt_file, "r", encoding='utf-8') as f:
        inputs = json.load(f)
    model_path = args.stable_model_path
    boxnet_path = args.boxnet_model_path
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    pipe = RegionallyT2IAdapterPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    if boxnet_path is not None:
        args.no_class = True
        pipe.boxnet, _, _ = build_model(args)
        pipe.boxnet.load_state_dict(torch.load(boxnet_path))
        pipe.boxnet = pipe.boxnet.to(device)
    
    pipe.enable_xformers_memory_efficient_attention()
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    torch.backends.cudnn.benchmark = True
    if args.seed == 0:
        args.seed = torch.randint(0, 1000000, (1,)).item()
    print(f"seed: {args.seed}")
    for rate in np.arange(0, 1.1, 0.1):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        for i, cur_input in enumerate(inputs):
            print(inputs[i]['prompt'])
            cur_path = os.path.join(save_path, "{}_{}".format(inputs[i]['phrases'][0], inputs[i]['phrases'][1]))
            os.makedirs(cur_path, exist_ok=True)
            # cur_input['original_boxes'] = None
            image = pipe( 
                data=cur_input, height=512, width=512, num_inference_steps=50, num_images_per_prompt=1, generator=generator, use_boxnet=True, 
                max_guidance_rate=rate).images[0]
            image.save(f"./images/{rate}.png")
''' 
Assistant: ”You are ChatGPT-4, a large language model trained by OpenAI. Your goal is to assist users by providing
helpful and relevant information. In this context, you are expected to identify prospective objects and generate specific coordinate box
locations for these objects in a description, considering their relative sizes and positions and the number of objects, such as ”six apples”.
The size of the image is 512*512.”

User: ”Provide box coordinates for an image with a cat in the middle of a car and a chair. Make the size of the
boxes as big as possible.”

Assistant: ”cat: (230, 196, 297, 301); car: (80, 270, 202, 352); chair: (341, 231, 447, 308)”

User: ”The sizes of objects do not reflect the object’ s relative sizes. The car’s box size is similar chair’s box size,
and the cat’s box size is quite similar to the car and chair. the car should be larger than the chair, and the chair
should be larger than the cat.”

Assistant: ......

User: ”The final result is a relatively coordinate position, so you need to divide the result by the width or height of the 
overall image and keep two decimal places. Arrange the results in the form of a python list and output 
them in order according to the position of the entity in the description.”

Assistant: ......

User: ”Provide box coordinates for an image with a solitary lighthouse perched on rugged cliffs 
illuminates the waves below. The horizon is painted in hues of pink and gold.”

Assistant: ......
'''

