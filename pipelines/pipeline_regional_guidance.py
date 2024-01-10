import math
from typing import Any, Callable, Dict, List, Optional, Union
import os
import inspect
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
    StableDiffusionAttendAndExcitePipeline,
)
from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_adapter import (StableDiffusionAdapterPipeline,
                                                                               StableDiffusionAdapterPipelineOutput,
                                                                               _preprocess_adapter_image)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from torch import einsum
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from utils.box_ops import box_cxcywh_to_xyxy
from .pipeline_freeofshow import register_attention_control_bbox, save_out_hook
from diffusers.models.attention_processor import Attention
from utils.utils import save_colored_mask, save_bbox_img, save_img_with_box

if is_xformers_available():
    import xformers


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
blocks = [0, 1, 2, 3]
kwargs = {}

class RegionT2I_AttnProcessor:
    def __init__(self, cross_attention_idx, attention_op=None):
        self.attention_op = attention_op
        self.cross_attention_idx = cross_attention_idx

    def region_rewrite(self, attn, hidden_states, query, region_list, height, width):

        def get_region_mask(region_list, feat_height, feat_width):
            exclusive_mask = torch.zeros((feat_height, feat_width))
            for region in region_list:
                start_h, start_w, end_h, end_w = region[-1]
                start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                    start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)
                exclusive_mask[start_h:end_h, start_w:end_w] += 1
            return exclusive_mask

        dtype = query.dtype
        seq_lens = query.shape[1]
        downscale = math.sqrt(height * width / seq_lens)

        # 0: context >=1: may be overlap
        feat_height, feat_width = int(height // downscale), int(width // downscale)
        region_mask = get_region_mask(region_list, feat_height, feat_width)

        query = rearrange(query, 'b (h w) c -> b h w c', h=feat_height, w=feat_width)
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b h w c', h=feat_height, w=feat_width)

        new_hidden_state = torch.zeros_like(hidden_states)
        new_hidden_state[:, region_mask == 0, :] = hidden_states[:, region_mask == 0, :]

        replace_ratio = 1.0
        new_hidden_state[:, region_mask != 0, :] = (1 - replace_ratio) * hidden_states[:, region_mask != 0, :]

        for region in region_list:
            region_key, region_value, region_box = region

            if attn.upcast_attention:
                query = query.float()
                region_key = region_key.float()

            start_h, start_w, end_h, end_w = region_box
            start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)

            attention_region = einsum('b h w c, b n c -> b h w n', query[:, start_h:end_h, start_w:end_w, :], region_key) * attn.scale
            if attn.upcast_softmax:
                attention_region = attention_region.float()

            attention_region = attention_region.softmax(dim=-1)
            attention_region = attention_region.to(dtype)

            hidden_state_region = einsum('b h w n, b n c -> b h w c', attention_region, region_value)
            new_hidden_state[:, start_h:end_h, start_w:end_w, :] += \
                replace_ratio * (hidden_state_region / (
                    region_mask.reshape(
                        1, *region_mask.shape, 1)[:, start_h:end_h, start_w:end_w, :]
                ).to(query.device))

        new_hidden_state = rearrange(new_hidden_state, 'b h w c -> b (h w) c')
        return new_hidden_state

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            is_cross = False
            encoder_hidden_states = hidden_states
        else:
            is_cross = True

            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:
                encoder_hidden_states = encoder_hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if is_xformers_available() and not is_cross:
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)

        if len(kwargs) > 0 and kwargs['guidance']:
            if is_cross:
                region_list = []
                for region in kwargs['region_list']:
                    region_key = attn.to_k(region[0])
                    region_value = attn.to_v(region[0])
                    region_key = attn.head_to_batch_dim(region_key)
                    region_value = attn.head_to_batch_dim(region_value)
                    region_list.append((region_key, region_value, region[1]))

                hidden_states = self.region_rewrite(
                    attn=attn,
                    hidden_states=hidden_states,
                    query=query,
                    region_list=region_list,
                    height=kwargs['height'],
                    width=kwargs['width'])

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

# for name in model.unet.attn_processors.keys():
def revise_regionally_attention_forward(unet):
    def change_forward(unet, count):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention':
                layer.set_processor(RegionT2I_AttnProcessor(count))
                if 'attn2' in name:
                    count += 1
            else:
                count = change_forward(layer, count)
        return count

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0)
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx)
    cross_attention_idx = change_forward(unet.up_blocks, cross_attention_idx)
    print(f'Number of attention layer registered {cross_attention_idx}')

def register_attention_control(unet):
        attn_procs = {}
        count = 0
        for name in unet.attn_processors.keys():
            if 'attn2' in name:
                count += 1
            attn_procs[name] = RegionT2I_AttnProcessor(count)
        unet.set_attn_processor(attn_procs)
        print(f'Number of attention layer registered {count}')



class RegionallyT2IAdapterPipeline(StableDiffusionPipeline):
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
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
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
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        # register_attention_control(self.unet)

    @torch.no_grad()
    def _encode_prompt(self, prompts, device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None):
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
    def  __call__(
        self,
        data,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        use_boxnet: bool = True,
        max_guidance_rate: int = 0.75,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.boxnet.eval()
        device = self._execution_device

        self.check_inputs(
            data["prompt"], height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        

        original_boxes = torch.tensor(data["original_boxes"]).to(device) if data["original_boxes"] is not None else None

        if use_boxnet is False and original_boxes is None:
            raise ValueError("Must provide original_boxes if do not use BoxNet.")
        
        if use_boxnet is False and max_guidance_rate>0:
            print("Warning: max_guidance_rate is useless if do not use BoxNet.")
        
        revise_regionally_attention_forward(self.unet)

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

        region_list = []

        words = data["prompt"].lower().split()
        # 修复代码bug
        region_prompts = [f"A photo of {' '.join(words[start:end+1])}" 
                          for start, end in data["entities"]]

        
        for region_prompt in region_prompts:
            region_emb = self._encode_prompt(
                region_prompt, device
            )
            region_list.append([region_emb, []])

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # predict the noise residual
                kwargs['guidance'] = False
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
                assert len(boxes) == len(region_list)
                for region, box in zip(region_list, boxes):
                    region[1] = box               
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                kwargs['region_list'] = region_list
                kwargs['height'] = height
                kwargs['width'] = width
                kwargs['guidance'] = True
                noise_pred = self.unet(latent_model_input, t,
                                    encoder_hidden_states=text_embeddings,
                                    # **cross_attention_kwargs
                                ).sample
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)


                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs).prev_sample
                                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        # for k, bboxes in enumerate(all_boxes):
        #     save_bbox_img(os.path.join(os.getcwd(), 'images'), bboxes, name=f"masked_bbox_{k}.png", device=device)

        if output_type == 'latent':
            image = latents
            has_nsfw_concept = None
        elif output_type == 'pil':
            # 8. Post-processing
            image = self.decode_latents(latents)
            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)
            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)
            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        if hasattr(self, 'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)