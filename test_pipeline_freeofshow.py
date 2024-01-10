import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import torch
import argparse
from pipelines.pipeline_freeofshow import FreeofShowPipeline
from utils.p2p import AttentionStore
from boxnet_models import build_model, add_boxnet_args
from utils.utils import save_colored_mask, save_bbox_img, save_img_with_box, preprocess_image
from PIL import Image

OUTPUT_DIR='./results'
BOXNET_MODEL_PATH='./ckpt/boxnet.pt'
STABLE_MODEL_PATH='/data/zsz/ssh/models/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', default='test_prompts.json', type=str,
                        help="Prompt file to generate images.")
    parser.add_argument('--stable_model_path', required=False,  default=STABLE_MODEL_PATH, 
                        type=str, help="Original stable diffusion model path.")
    parser.add_argument('--boxnet_model_path', required=False,  default=BOXNET_MODEL_PATH,
                        type=str, help="BoxNet model path.")
    parser.add_argument('--output_dir', required=False, type=str,  default=OUTPUT_DIR,
                        help="Output dir for results.")
    parser.add_argument('--seed', default=0, type=int,
                        help="Random seed.")
    parser.add_argument('--max_guidance_rate', default=0.75, type=float,
                        help="The rate using boxnet prediction.")
    parser.add_argument('--mask_mode', default='gaussin_zero_one', type=str, choices=['gaussin_zero_one', 'zero_one'],
                        help="mask mode.")
    parser.add_argument('--soft_mask_rate', default=0.2, type=float,
                        help="Soft mask rate for self mask.")
    parser.add_argument('--focus_rate', default=1.0, type=int,
                        help="Focus rate on area in-box")
    parser = add_boxnet_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda")
    with open(args.prompt_file, "r", encoding='utf-8') as f:
        inputs = json.load(f)
    model_path = args.stable_model_path
    boxnet_path = args.boxnet_model_path
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    pipe = FreeofShowPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
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
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for i, cur_input in enumerate(inputs):
        print(inputs[i]['prompt'])
        cur_path = os.path.join(save_path, "{}_{}".format(inputs[i]['phrases'][0], inputs[i]['phrases'][1]))
        os.makedirs(cur_path, exist_ok=True)
        controller = AttentionStore()
        cur_input['original_boxes'] = None
        ########################################
        # cur_input['prompt'] = "a cat and a dog"
        # init_latents = Image.open('./catdog.jpg').convert('RGB')
        # image = preprocess_image(init_latents).to(device)
        # init_latent_dist = pipe.vae.encode(image.half()).latent_dist
        # init_latents = init_latent_dist.sample(generator=generator)
        # init_latents = 0.18215 * init_latents
        # shape = init_latents.shape
        # noise = torch.randn(shape, generator=generator, device=device) # if len(noise_preds) == 0 else noise_preds[time]
        # pipe.scheduler.set_timesteps(5)
        # timesteps = pipe.scheduler.timesteps.to(device)
        # step = 3
        # t = timesteps[step - 1] * torch.ones(init_latents.shape[0]).to(device)
        # t = t.long()
        # latents = pipe.scheduler.add_noise(init_latents, noise, t)
        ###########################################
        images, all_step_bboxes, attn_img = pipe.log_imgs( 
            cur_input, num_inference_steps=50, num_images_per_prompt=1, generator=generator, controller=controller, mask_control=True, mask_self=False, use_boxnet=True, 
            mask_mode=args.mask_mode, soft_mask_rate=args.soft_mask_rate, focus_rate=args.focus_rate, max_guidance_rate=args.max_guidance_rate)

        images[0].save(os.path.join(cur_path, f"masked_result.jpg"))
        # print(all_step_bboxes)
        for k, bboxes in enumerate(all_step_bboxes):
            save_bbox_img(cur_path, bboxes, name=f"masked_bbox_{k}.png", device=device)
        attn_img.save(os.path.join(cur_path, f"masked_attn.jpg"))


''' 
Assistant: ”You are ChatGPT-4, a large language model trained by OpenAI. Your goal is to assist users by providing
helpful and relevant information. In this context, you are expected to identify prospective objects and generate specific coordinate box
locations for these objects in a description, considering their relative sizes and positions and the number of objects, such as ”six apples”.
The size of the image is 512*512.”

User: ”Provide box coordinates for an image with a cat in the middle of a car and a chair. Make the size of the
boxes as big as possible.”

Assistant: ......

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

