import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
import torch
import inspect
from typing import Callable, List, Optional, Union
import xformers
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor

from utils.utils import numpy_to_pil
from utils.box_ops import box_cxcywh_to_xyxy
from utils.p2p import AttentionStore, show_cross_attention_blackwhite, show_cross_attention, EmptyControl
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_adapter import (StableDiffusionAdapterPipeline,
                                                                               StableDiffusionAdapterPipelineOutput,
                                                                               _preprocess_adapter_image)
from utils.utils import save_colored_mask, save_bbox_img, save_img_with_box

blocks = [0, 1, 2, 3]

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f for f in features if f is not None and isinstance(
            f, torch.Tensor)]  # .float() .detach()
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f for k, f in features.items()}  # .float()
        setattr(module, name, features)
    else:
        setattr(module, name, features)  # .float()


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out


def build_normal(u_x, u_y, d_x, d_y, step, device):
    x, y = torch.meshgrid(torch.linspace(0,1,step), torch.linspace(0,1,step))
    x = x.to(device)
    y = y.to(device)
    out_prob = (1/2/torch.pi/d_x/d_y)*torch.exp(-1/2*(torch.square((x-u_x)/d_x)+torch.square((y-u_y)/d_y)))
    return out_prob

def uniq_masks(all_masks, zero_masks=None, scale=1.0):
    uniq_masks = torch.stack(all_masks)
    # num = all_masks.shape[0]
    uniq_mask = torch.argmax(uniq_masks, dim=0)
    if zero_masks is None:
        all_masks = [((uniq_mask==i)*mask*scale).float().clamp(0, 1.0) for i, mask in enumerate(all_masks)]
    else:
        all_masks = [((uniq_mask==i)*mask*scale).float().clamp(0, 1.0) for i, mask in enumerate(zero_masks)]

    return all_masks


def build_masks(bboxes, size, mask_mode="gaussin_zero_one", focus_rate=1.0):
    all_masks = []
    zero_masks = []
    for bbox in bboxes:
        x0,y0,x1,y1 = bbox
        mask = build_normal((y0+y1)/2, (x0+x1)/2, (y1-y0)/4, (x1-x0)/4, size, bbox.device)
        zero_mask = torch.zeros_like(mask)
        zero_mask[int(y0 * size):min(int(y1 * size)+1, size), int(x0 * size):min(int(x1 * size)+1, size)] = 1.0
        zero_masks.append(zero_mask)
        all_masks.append(mask)
    if mask_mode == 'zero_one':
        return zero_masks
    elif mask_mode == 'guassin':
        all_masks = uniq_masks(all_masks, scale=focus_rate)
        return all_masks
    elif mask_mode == 'gaussin_zero_one':
        all_masks = uniq_masks(all_masks, zero_masks, scale=focus_rate)
        return all_masks
    else:
        raise ValueError("Not supported mask_mode.")


class BboxCrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet, bboxes, entity_indexes, mask_control, mask_self, with_uncond, mask_mode, soft_mask_rate, focus_rate):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.bboxes = bboxes
        self.entity_indexes = entity_indexes
        self.mask_control = mask_control
        self.mask_self = mask_self
        self.with_uncond = with_uncond
        self.mask_mode = mask_mode
        self.soft_mask_rate = soft_mask_rate
        self.focus_rate = focus_rate

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if self.with_uncond:
            cond_attention_probs = attention_probs[batch_size//2:]
        else:
            cond_attention_probs = attention_probs

        if self.mask_control and np.sqrt(cond_attention_probs.shape[1]) == 16:
            
            if is_cross:
                size = int(np.sqrt(sequence_length))
                all_masks = build_masks(self.bboxes, size, mask_mode=self.mask_mode, focus_rate=self.focus_rate)
                for pos, mask in zip(self.entity_indexes, all_masks):
                    start = pos[0]
                    end = pos[-1]
                    if mask.sum() <= 0:  # sequence_length *  0.004:
                        continue
                    mask = mask.reshape((sequence_length, -1)).to(hidden_states.device)
                    mask = mask.expand(-1, (end-start+1))
                    cond_attention_probs[:, :, start+1:end+2] = cond_attention_probs[:, :, start+1:end+2] * mask
            elif self.mask_self:
                size = int(np.sqrt(sequence_length))
                # must be 1/0
                all_masks = build_masks(self.bboxes, size, mask_mode=self.mask_mode, focus_rate=self.focus_rate)
                for img_mask in all_masks:
                    if img_mask.sum() <= 0:  # sequence_length *  0.004:
                        continue
                    img_mask = img_mask.reshape(sequence_length)
                    mask_index = img_mask.nonzero().squeeze(-1)
                    mask = torch.ones(sequence_length, sequence_length).to(hidden_states.device)

                    mask[:, mask_index] = mask[:, mask_index] * img_mask.unsqueeze(-1)
                    cond_attention_probs = cond_attention_probs * mask + cond_attention_probs * (1-mask) * self.soft_mask_rate
            if self.with_uncond:
                attention_probs[batch_size//2:] = cond_attention_probs
            else:
                attention_probs = cond_attention_probs

        self.attnstore(cond_attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states




def register_attention_control_bbox(model, controller, bboxes, entity_indexes, mask_control=False, mask_self=False, 
                                    with_uncond=False, mask_mode='gaussin_zero_one', soft_mask_rate=0.2, focus_rate=1.0):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = BboxCrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet, bboxes=bboxes, entity_indexes=entity_indexes, 
            mask_control=mask_control, mask_self=mask_self, with_uncond=with_uncond, mask_mode=mask_mode,
            soft_mask_rate=soft_mask_rate, focus_rate=focus_rate
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


class FreeofShowPipeline(StableDiffusionPipeline):

    def __init__(
        self, 
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )
        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #     model_path, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(
        #     os.path.join(model_path, "text_encoder"))
        # self.vae = AutoencoderKL.from_pretrained(
        #     model_path, subfolder="vae")
        # self.unet = UNet2DConditionModel.from_pretrained(
        #     model_path, subfolder="unet")
        # self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
        #     model_path, subfolder="scheduler")
        
        self.boxnet = None
        save_hook = save_out_hook
        self.feature_blocks = []
        for idx, block in enumerate(self.unet.down_blocks):
            if idx in blocks:
                block.register_forward_hook(save_hook)
                self.feature_blocks.append(block)

        for idx, block in enumerate(self.unet.up_blocks):
            if idx in blocks:
                block.register_forward_hook(save_hook)
                self.feature_blocks.append(block)

    @torch.no_grad()
    def _encode_prompt(self, prompts, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompts) if isinstance(prompts, list) else 1
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        # pad_index = self.tokenizer.vocab['[PAD]']
        # attention_mask = text_input_ids.ne(pad_index).type(self.text_encoder.embeddings.word_embeddings.weight.dtype).to(device)

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
        )
        text_embeddings = text_embeddings[0]
        # print("text_embeddings: ")
        # print(text_embeddings)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompts) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompts)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompts} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]

            uncond_text_inputs = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_text_inputs.input_ids

            uncond_embeddings = self.text_encoder(
                # uncond_input.input_ids.to(device),
                uncond_input_ids.to(device),
                # attention_mask=uncond_attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(
            self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def get_predicted_boxes(self, feature_blocks, latents, noise_pred_text, cat_embeddings, box_num, t, device):
        activations = []
        for block in feature_blocks:
            activations.append(block.activations)
            block.activations = None

        activations = [activations[0][0], activations[1][0], activations[2][0], activations[3][0], activations[4], activations[5], activations[6], activations[7]]

        activations = activations[-4:]

        assert all([isinstance(acts, torch.Tensor) for acts in activations])
        size = latents.shape[2:]
        resized_activations = []
        for acts in activations:
            acts = torch.nn.functional.interpolate(
                acts, size=size, mode="bilinear"
            )
            resized_activations.append(acts)
        
        features =  torch.cat(resized_activations, dim=1)

        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[t]).to(device) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noise_pred_text.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noise_level = sqrt_one_minus_alpha_prod * noise_pred_text
        outputs = self.boxnet(features, noise_level, queries=cat_embeddings) 
        out_bbox = outputs['pred_boxes']
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes[0][:box_num]
        return boxes


    @torch.no_grad()
    def log_imgs(
        self,
        data,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        controller: Optional[AttentionStore] = None,
        mask_control: bool = False,
        mask_self: bool = True,
        use_boxnet: bool = True,
        mask_mode: Optional[str] = 'gaussin_zero_one',
        soft_mask_rate: int = 0.2,
        focus_rate: int = 1.0,
        max_guidance_rate: int = 0.75,
        **kwargs
    ):
        # self.boxnet.eval()
        device = self._execution_device

        original_boxes = torch.tensor(data["original_boxes"]).to(device) if data["original_boxes"] is not None else None

        if use_boxnet is False and original_boxes is None:
            raise ValueError("Must provide original_boxes if do not use BoxNet.")
        
        if use_boxnet is False and max_guidance_rate>0:
            print("Warning: max_guidance_rate is useless if do not use BoxNet.")

        feature_blocks = []
        for idx, block in enumerate(self.unet.down_blocks):
            if idx in blocks:
                block.register_forward_hook(save_out_hook)
                feature_blocks.append(block) 
                
        for idx, block in enumerate(self.unet.up_blocks):
            if idx in blocks:
                block.register_forward_hook(save_out_hook)
                feature_blocks.append(block)  

        prompts = []
        cat_embeddings = []
        prompts.append(data["prompt"])
        cat_input_id = self.tokenizer(
            data["phrases"],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        tmp_embed = self.text_encoder(cat_input_id)[1]
        cat_embed = torch.zeros(30, 768).to(device)
        cat_embed[:len(data["phrases"])] = tmp_embed
        cat_embeddings = cat_embed.unsqueeze(0)
        box_num = len(data["phrases"])
        entities = data["entities"]

        batch_size = 1 if isinstance(prompts, str) else len(prompts)
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompts, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps


        if latents is None:
            shape = (batch_size * num_images_per_prompt,
                     self.unet.in_channels, height // 8, width // 8)
            latents = torch.randn(shape, generator=generator,
                                  device=device, dtype=text_embeddings.dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma

        latents = latents.to(self.unet.dtype)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        all_boxes = []
        for i, t in enumerate(tqdm(timesteps)):
            if controller is not None:
                register_attention_control_bbox(self, controller, None, None, False, False, False, mask_mode)

            # predict the noise residual
            noise_pred_text = self.unet(latents, t,
                                   encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            ################################################################
            # if original_boxes is None, we will use boxnet to predict boxes all the time.
            if use_boxnet and (original_boxes is None or i >= num_inference_steps * (1 - max_guidance_rate)):
                boxes = self.get_predicted_boxes(feature_blocks, latents, noise_pred_text, cat_embeddings, box_num, t, device)
            else:
                boxes = original_boxes
            if i % 5 == 0:
                all_boxes.append(boxes)
            ###############################################################
            # for k, bboxes in enumerate(all_boxes):
            #     save_bbox_img(os.getcwd(), bboxes, name=f"masked_bbox_{k}.png", device=device)
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            
            if controller is not None:
                register_attention_control_bbox(self, controller, boxes, entities, mask_control=mask_control, mask_self=mask_self, 
                                                with_uncond=True, mask_mode=mask_mode, soft_mask_rate=soft_mask_rate, focus_rate=focus_rate)

            noise_pred = self.unet(latent_model_input, t,
                                   encoder_hidden_states=text_embeddings).sample
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)
            if controller is not None:
                latents = controller.step_callback(latents)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        if controller is not None:
            _, attn_img = show_cross_attention(
                prompts, self.tokenizer, controller, res=16, from_where=("up", "down"), save_img=False)

        return image, all_boxes, attn_img


    def __call__(
        self
    ):
        pass