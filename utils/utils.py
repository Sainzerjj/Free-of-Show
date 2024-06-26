import os
import json
from collections import OrderedDict
import torch
# from cn_clip.clip.configuration_bert import BertConfig
# from cn_clip.clip.modeling_bert import BertModel
from typing import Union, List
from PIL import Image, ImageDraw
from torch import nn
import imgviz
import numpy as np
from diffusers.utils import PIL_INTERPOLATION

CONFIG_NAME = "RoBERTa-wwm-ext-large-chinese.json"
WEIGHT_NAME = "pytorch_model.bin"

def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = acc * 100

    return acc

def resize_and_concatenate(activations: List[torch.Tensor], reference):
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = reference.shape[2:]
    resized_activations = []
    for acts in activations:
        acts = nn.functional.interpolate(
            acts, size=size, mode="bilinear"
        )
        acts = acts[:1]
        acts = acts.transpose(1,3)
        resized_activations.append(acts)
    
    return torch.cat(resized_activations, dim=3)

def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images


def save_colored_mask(save_path, mask_pil):
    """保存调色板彩色图"""
    lbl_pil = mask_pil.convert('P')
    # lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    colormap = imgviz.label_colormap(80)
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def save_bbox_img(colored_res_path, bboxes, size=512, name="bbox.png", device=torch.device("cuda")):
    scale_fct = torch.tensor([size, size, size, size]).to(device)
    bboxes = bboxes * scale_fct
    out_image = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(out_image)
    for n, b in enumerate(bboxes):

        draw.rectangle(((b[0], b[1]), (b[2], b[3])),
                       fill=None, outline=n+1, width=5)
    save_colored_mask(os.path.join(colored_res_path, name), out_image)
    return out_image

def save_img_with_box(img, bboxes, device=torch.device("cuda")):
    W, H = img.size
    scale_fct = torch.tensor([W, H, W, H]).to(device)
    bboxes = bboxes * scale_fct
    colors = ['red', 'green']
    draw = ImageDraw.Draw(img)
    for n, b in enumerate(bboxes):
        draw.rectangle(((b[0], b[1]),(b[2], b[3])), fill=None, outline=colors[n], width=5)
    # save_colored_mask(os.path.join(colored_res_path, name), attn_img)
    return img



def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def count_x0_batch(
        scheduler,
        model_output: torch.FloatTensor,
        timesteps: List[int],
        sample: torch.FloatTensor,
        clip_sample: bool = False,
    ):
    # for t in timesteps:
    # t = timestep
        # 1. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timesteps].resize(len(timesteps), 1, 1, 1).expand(-1, model_output.shape[1], model_output.shape[2], model_output.shape[3])
    # alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
    beta_prod_t = 1 - alpha_prod_t
    # beta_prod_t_prev = 1 - alpha_prod_t_prev

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # Clip "predicted x_0"
    if clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    # pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * scheduler.betas[t]) / beta_prod_t
    # current_sample_coeff = scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

    # # 5. Compute predicted previous sample µ_t
    # # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    # pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
    
    return pred_original_sample


def count_x_0(
        scheduler,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        clip_sample: bool = False,
    ):
    t = timestep
    # 1. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # Clip "predicted x_0"
    if clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * scheduler.betas[t]) / beta_prod_t
    current_sample_coeff = scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    # # 6. Add noise
    # variance = 0
    # if t > 0:
    #     device = model_output.device
    #     if device.type == "mps":
    #         # randn does not work reproducibly on mps
    #         variance_noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator)
    #         variance_noise = variance_noise.to(device)
    #     else:
    #         variance_noise = torch.randn(
    #             model_output.shape, generator=generator, device=device, dtype=model_output.dtype
    #         )
    #     if self.variance_type == "fixed_small_log":
    #         variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
    #     else:
    #         variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

    # pred_prev_sample = pred_prev_sample + variance
    
    return pred_original_sample, pred_prev_sample


def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images


def check_dir(save_directory):
    if os.path.isfile(save_directory):
        print(f"Provided path ({save_directory}) should be a directory, not a file")
        return

    os.makedirs(save_directory, exist_ok=True)


def save_images(images, save_directory, prompts, repeat=1):
    check_dir(save_directory)
    width, height = images[0].size
    assert len(images) == len(prompts) * repeat, "Input images has wrong number."
    for i in range(0, len(images), repeat):
        new_img = Image.new("RGB", (width*repeat, height))
        for j in range(repeat):
            new_img.paste(images[i+j], (width*j, 0))
        new_img.save(os.path.join(save_directory, "{}.png".format(prompts[int(i/repeat)])))

def save_config(bert_config, save_directory):
    check_dir(save_directory)
    # print(bert_config)
    dict_config = {
        "vocab_size": bert_config.vocab_size,
        "hidden_size": bert_config.hidden_size,
        "num_hidden_layers": bert_config.num_hidden_layers,
        "num_attention_heads": bert_config.num_attention_heads,
        "intermediate_size": bert_config.intermediate_size,
        "hidden_act": bert_config.hidden_act,
        "hidden_dropout_prob": bert_config.hidden_dropout_prob,
        "attention_probs_dropout_prob": bert_config.attention_probs_dropout_prob,
        "max_position_embeddings": bert_config.max_position_embeddings,
        "type_vocab_size": bert_config.type_vocab_size,
        "initializer_range": bert_config.initializer_range,
    }
    with open(os.path.join(save_directory, CONFIG_NAME), 'w', encoding='utf-8') as f:
        json.dump(dict_config, f, indent=4)


def save_model(model, save_directory):
    check_dir(save_directory)
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(save_directory, WEIGHT_NAME))


# def load_config(from_pretrained):
#     with open(os.path.join(from_pretrained, CONFIG_NAME), 'r', encoding='utf-8') as f:
#         config = json.load(f)

#     bert_config = BertConfig(
#         vocab_size_or_config_json_file=config["vocab_size"],
#         hidden_size=config["hidden_size"],
#         num_hidden_layers=config["num_hidden_layers"],
#         num_attention_heads=config["num_attention_heads"],
#         intermediate_size=config["intermediate_size"],
#         hidden_act=config["hidden_act"],
#         hidden_dropout_prob=config["hidden_dropout_prob"],
#         attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
#         max_position_embeddings=config["max_position_embeddings"],
#         type_vocab_size=config["type_vocab_size"],
#         initializer_range=config["initializer_range"],
#         layer_norm_eps=1e-12,
#     )
#     return bert_config


# def load_clip(from_pretrained, bert_config, use_fp16=False):
#     # bert_config = load_config(from_pretrained)
#     bert_model = BertModel(bert_config)
    
#     with open(os.path.join(from_pretrained, WEIGHT_NAME), 'rb') as opened_file:
#         # loading saved checkpoint
#         checkpoint = torch.load(opened_file, map_location="cpu")
#     if "state_dict" in checkpoint:
#         sd = checkpoint["state_dict"]
#     else:
#         sd = checkpoint
#     if next(iter(sd.items()))[0].startswith('module'):
#         sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
#     new_sd = OrderedDict()
#     for key in sd:
#         if key.startswith('bert'):
#             new_sd[key[len('bert.'):]] = sd[key]
#     if not new_sd:
#         new_sd = sd
#     print("load clip model ckpt from {}".format(os.path.join(from_pretrained, WEIGHT_NAME)))
#     bert_model.load_state_dict(new_sd, strict=True)
#     # bert_model = bert_model.to(device)
#     if use_fp16:
#         bert_model = bert_model.half()
#     return bert_model


def tokenize(tokenizer, texts: Union[str, List[str]], context_length: int = 64) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 24 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([tokenizer.vocab['[CLS]']] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))[
                                                        :context_length - 2] + [tokenizer.vocab['[SEP]']])

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result






