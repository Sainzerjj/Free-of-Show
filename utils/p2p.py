import abc
import os
from typing import Optional, Union, Tuple, List, Callable, Dict

import torch
import numpy as np
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import cv2

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "/usr/share/fonts/truetype/tlwg/TlwgTypo-Bold.ttf", textSize, encoding="utf-8")
    # fontText = cv2.FONT_HERSHEY_SIMPLEX
    draw.text((left, top), text, textColor, font=fontText)
    return np.asarray(img)


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    # img[:h] = image
    # img = cv2ImgAddText(img, text, int(w*.4), h+int(offset/4), textColor=text_color, textSize=int(offset/2))
    # return img
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, fontScale=1, thickness=2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def display(image, save_dir='tmp'):
    # save_dir = 'tmp'
    exist_num = 0
    save_file = os.path.join(save_dir, f"save_{exist_num}.jpg")
    while os.path.isfile(save_file):
        exist_num += 1
        save_file = os.path.join(save_dir, f"save_{exist_num}.jpg")
    image.save(save_file)


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                save_img: bool = True) -> Image.Image:
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if save_img:
        display(pil_img)
    # pil_imgs = [Image.fromarray(image) for image in images]
    return pil_img


def show_cross_attention_blackwhite(prompts, tokenizer, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, save_img: bool = True):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    out_images = []
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        out_images.append(image)
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    pil_image = view_images(np.stack(images, axis=0), save_img = save_img)
    return attention_maps, pil_image


def show_cross_attention(prompts,
                         tokenizer,
                         attention_store: AttentionStore,
                         res: int,
                         from_where: List[str],
                         select: int = 0, 
                         save_img: bool = True
                         ):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    out_imgs = []
    white_image = Image.new('RGB', (500, 500), (255, 255, 255))
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = show_image_relevance(image, white_image)
        image = image.astype(np.uint8)        
        image = np.array(Image.fromarray(image).resize((256, 256)))
        out_imgs.append(image)
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    pil_img=view_images(np.stack(images, axis=0), save_img=save_img)
    return attention_maps, pil_img


def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min()+1e-8)
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def mean_iou(seg_image_path, attn_image_path, Threshold=0.3):
    seg_image = Image.open(seg_image_path)
    seg_img_data = np.asarray(seg_image).astype(bool)
    seg_img_size=seg_image.size

    if len(seg_img_data.shape) == 3:
        seg_img_data=seg_img_data[:,:,-1]

    image = Image.open(attn_image_path)
    image=image.resize(seg_img_size)
    img_data = np.asarray(image).astype(np.float16)

    if len(img_data.shape) == 3:
        img_data=img_data[:,:,-1]

    img_data /=img_data.max()
    img_mask = img_data > Threshold

    intersection = np.logical_and(seg_img_data, img_mask).sum()
    union = np.logical_or(seg_img_data, img_mask).sum()

    return float(intersection/union)