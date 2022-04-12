from typing import List, Dict, Union, Tuple

from PIL import Image, ImageDraw, ImageFilter
import spacy
import hashlib
import os

import torch
import torchvision
import torchvision.transforms as transforms
import clip
from transformers import BertTokenizer, RobertaTokenizerFast
import ruamel.yaml as yaml

from interpreter import Box
from albef.model import ALBEF
from albef.utils import *
from albef.vit import interpolate_pos_embed

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

class Executor:
    def __init__(self, device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None) -> None:
        IMPLEMENTED_METHODS = ["crop", "blur", "shade"]
        if any(m not in IMPLEMENTED_METHODS for m in box_representation_method.split(",")):
            raise NotImplementedError
        IMPLEMENTED_AGGREGATORS = ["max", "sum"]
        if method_aggregator not in IMPLEMENTED_AGGREGATORS:
            raise NotImplementedError
        self.box_representation_method = box_representation_method
        self.method_aggregator = method_aggregator
        self.enlarge_boxes = enlarge_boxes
        self.device = device
        self.expand_position_embedding = expand_position_embedding
        self.square_size = square_size
        self.blur_std_dev = blur_std_dev
        self.cache_path = cache_path

    def preprocess_image(self, image: Image) -> List[torch.Tensor]:
        return [preprocess(image) for preprocess in self.preprocesses]
   
    def preprocess_text(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def tensorize_inputs(self, caption: str, image: Image, boxes: List[Box], image_name: str = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        images = []
        for preprocess in self.preprocesses:
            images.append([])
        if self.cache_path is None or any([not os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")) for model_name in self.model_names for method_name in self.box_representation_method.split(',')]):
            if "crop" in self.box_representation_method:
                for i in range(len(boxes)):
                    image_i = image.copy()
                    box = [
                        max(boxes[i].left-self.enlarge_boxes, 0),
                        max(boxes[i].top-self.enlarge_boxes, 0),
                        min(boxes[i].right+self.enlarge_boxes, image_i.width),
                        min(boxes[i].bottom+self.enlarge_boxes, image_i.height)
                    ]
                    image_i = image_i.crop(box)
                    preprocessed_images = self.preprocess_image(image_i)
                    for j, img in enumerate(preprocessed_images):
                        images[j].append(img.to(self.device))
            if "blur" in self.box_representation_method:
                for i in range(len(boxes)):
                    image_i = image.copy()
                    mask = Image.new('L', image_i.size, 0)
                    draw = ImageDraw.Draw(mask)
                    box = (
                        max(boxes[i].left-self.enlarge_boxes, 0),
                        max(boxes[i].top-self.enlarge_boxes, 0),
                        min(boxes[i].right+self.enlarge_boxes, image_i.width),
                        min(boxes[i].bottom+self.enlarge_boxes, image_i.height)
                    )
                    draw.rectangle([box[:2], box[2:]], fill=255)
                    blurred = image_i.filter(ImageFilter.GaussianBlur(self.blur_std_dev))
                    blurred.paste(image_i, mask=mask)
                    preprocessed_images = self.preprocess_image(blurred)
                    for j, img in enumerate(preprocessed_images):
                        images[j].append(img.to(self.device))
            if "shade" in self.box_representation_method:
                for i in range(len(boxes)):
                    TINT_COLOR = (240, 0, 30)
                    image_i = image.copy().convert('RGBA')
                    overlay = Image.new('RGBA', image_i.size, TINT_COLOR+(0,))
                    draw = ImageDraw.Draw(overlay)
                    box = [
                        max(boxes[i].left-self.enlarge_boxes, 0),
                        max(boxes[i].top-self.enlarge_boxes, 0),
                        min(boxes[i].right+self.enlarge_boxes, image_i.width),
                        min(boxes[i].bottom+self.enlarge_boxes, image_i.height)
                    ]
                    draw.rectangle((tuple(box[:2]), tuple(box[2:])), fill=TINT_COLOR+(127,))
                    shaded_image = Image.alpha_composite(image_i, overlay)
                    shaded_image = shaded_image.convert('RGB')
                    preprocessed_images = self.preprocess_image(shaded_image) # []
                    # for model in self.models:
                    #     preprocessed_images.append(preprocess(shaded_image))
                    for j, img in enumerate(preprocessed_images):
                        images[j].append(img.to(self.device))
            imgs = [torch.stack(image_list) for image_list in images]
        else:
            imgs = [[] for _ in self.models]
        text_tensor = self.preprocess_text(caption.lower()).to(self.device)
        return imgs, text_tensor

    @torch.no_grad()
    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None) -> torch.Tensor:
        images, text_tensor = self.tensorize_inputs(caption, image, boxes, image_name)
        all_logits_per_image = []
        all_logits_per_text = []
        box_representation_methods = self.box_representation_method.split(',')
        caption_hash = hashlib.md5(caption.encode('utf-8')).hexdigest()
        for model, images_t, model_name in zip(self.models, images, self.model_names):
            if self.cache_path is not None:
                text_cache_path = os.path.join(self.cache_path, model_name, "text"+("_shade" if self.box_representation_method == "shade" else ""))
            image_features = None
            text_features = None
            if self.cache_path is not None and os.path.exists(os.path.join(self.cache_path, model_name)):
                if os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")):
                    text_features = torch.load(os.path.join(text_cache_path, caption_hash+".pt"), map_location=self.device)
                if os.path.exists(os.path.join(self.cache_path, model_name, image_name)):
                    if all([os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")) for method_name in box_representation_methods]):
                        image_features = []
                        for method_name in box_representation_methods:
                            features = torch.load(os.path.join(self.cache_path, model_name, image_name, method_name+".pt"), map_location=self.device)
                            image_features.append(torch.stack([
                                features[(box.x, box.y, box.w, box.h)]
                                for box in boxes
                            ]))
                        image_features = torch.stack(image_features)
                        image_features = image_features.view(-1, image_features.shape[-1])
            logits_per_image, logits_per_text, image_features, text_features = self.call_model(model, images_t, text_tensor, image_features=image_features, text_features=text_features)
            all_logits_per_image.append(logits_per_image)
            all_logits_per_text.append(logits_per_text)
            if self.cache_path is not None and image_name is not None and image_features is not None:
                image_features = image_features.view(len(box_representation_methods), len(boxes), image_features.shape[-1])
                if not os.path.exists(os.path.join(self.cache_path, model_name, image_name)):
                    os.makedirs(os.path.join(self.cache_path, model_name, image_name))
                for i in range(image_features.shape[0]):
                    method_name = box_representation_methods[i]
                    if not os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")):
                        image_features_dict = {(box.x, box.y, box.w, box.h): image_features[i,j,:].cpu() for j, box in enumerate(boxes)}
                        torch.save(image_features_dict, os.path.join(self.cache_path, model_name, image_name, method_name+".pt"))
            if self.cache_path is not None and not os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")) and text_features is not None:
                assert text_features.shape[0] == 1
                if not os.path.exists(text_cache_path):
                    os.makedirs(text_cache_path)
                torch.save(text_features.cpu(), os.path.join(text_cache_path, caption_hash+".pt"))

        all_logits_per_image = torch.stack(all_logits_per_image).sum(0)
        all_logits_per_text = torch.stack(all_logits_per_text).sum(0)
        if self.method_aggregator == "max":
            all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).max(dim=0, keepdim=True)[0]
        elif self.method_aggregator == "sum":
            all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).sum(dim=0, keepdim=True)
        return all_logits_per_text.view(-1)

class ClipExecutor(Executor):
    def __init__(self, clip_model: str = "ViT-B/32", device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None) -> None:
        super().__init__(device, box_representation_method, method_aggregator, enlarge_boxes, expand_position_embedding, square_size, blur_std_dev, cache_path)
        self.clip_models = clip_model.split(",")
        self.model_names = [model_name.replace("/", "_") for model_name in self.clip_models]
        self.models = []
        self.preprocesses = []
        for model_name in self.clip_models:
            model, preprocess = clip.load(model_name, device=device, jit=False)
            self.models.append(model)
            if self.square_size:
                print("Square size!")
                preprocess.transforms[0] = transforms.Resize((model.visual.input_resolution, model.visual.input_resolution), interpolation=transforms.InterpolationMode.BICUBIC)
            self.preprocesses.append(preprocess)
        self.models = torch.nn.ModuleList(self.models)

    def preprocess_text(self, text: str) -> torch.Tensor:
        if "shade" in self.box_representation_method:
            return clip.tokenize([text.lower()+" is in red color."])
        return clip.tokenize(["a photo of "+text.lower()])

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: torch.Tensor, image_features: torch.Tensor = None, text_features: torch.Tensor = None) -> torch.Tensor:
        if image_features is None:
            print('computing image features')
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if text_features is None:
            print('computing text features')
            text_features = model.encode_text(text)
            # normalized features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, image_features, text_features

    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None) -> torch.Tensor:
        if self.expand_position_embedding:
            original_preprocesses = self.preprocesses
            new_preprocesses = []
            original_position_embeddings = []
            for model_name, model, preprocess in zip(self.clip_models, self.models, self.preprocesses):
                if "RN" in model_name:
                    model_spatial_dim = int((model.visual.attnpool.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.attnpool.positional_embedding.clone()
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.attnpool.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.attnpool.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                else:
                    model_spatial_dim = int((model.visual.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.positional_embedding.clone()
                    model.visual.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                new_preprocesses.append(transform)
                original_position_embeddings.append(original_positional_embedding)
            self.preprocesses = new_preprocesses
        result = super().__call__(caption, image, boxes, image_name)
        if self.expand_position_embedding:
            self.preprocesses = original_preprocesses
            for model, model_name, pos_embedding in zip(self.models, self.clip_models, original_position_embeddings):
                if "RN" in model_name:
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(pos_embedding)
                else:
                    model.visual.positional_embedding = torch.nn.Parameter(pos_embedding)
        return result

class ClipGradcamExecutor(ClipExecutor):
    def __init__(self, clip_model: str = "ViT-B/32", device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", gradcam_alpha: List[float] = [1.0], expand_position_embedding: bool = False, background_subtract: bool = False, square_size: bool = False, blur_std_dev: int = 100, gradcam_ensemble_before: bool = False) -> None:
        super().__init__(clip_model, device, box_representation_method, method_aggregator, False, expand_position_embedding, square_size, blur_std_dev, None)
        self.clip_models = clip_model.split(",")
        for i in range(len(self.clip_models)):
            if "ViT" in self.clip_models[i]:
                import clip_mm_explain
                self.models[i] = clip_mm_explain.load(self.clip_models[i], device=device, jit=False)[0]
        self.gradcam_alpha = gradcam_alpha
        self.expand_position_embedding = expand_position_embedding
        self.background_subtract = background_subtract
        self.gradcam_ensemble_before = gradcam_ensemble_before

    def __call__(self, caption: str, image: Image, boxes: List[Box], return_gradcam=False, image_name: str = None) -> torch.Tensor:
        if self.background_subtract:
            self.background_subtract = False
            background = self("", image, boxes, True)
            self.background_subtract = True
        text_tensor = self.preprocess_text(caption).to(self.device)
        scores_list = []
        gradcam_list = []
        # print(len(self.clip_models), len(self.models), len(self.preprocesses), len(self.gradcam_alpha))
        for model_name, model, preprocess, gradcam_alpha in zip(self.clip_models, self.models, self.preprocesses, self.gradcam_alpha):
            if "RN" in model_name:
                if self.expand_position_embedding:
                    model_spatial_dim = int((model.visual.attnpool.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.attnpool.positional_embedding.clone()
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.attnpool.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.attnpool.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                    image_t = transform(image).unsqueeze(0).to(self.device)
                    print(model.visual.attnpool.positional_embedding.shape, image_t.shape, model_spatial_dim, patch_size, image.size)
                else:
                    image_t = preprocess(image).unsqueeze(0).to(self.device)
                activations_and_grads = ActivationsAndGradients(model, [model.visual.layer4], None)
                height_width_ratio = image_t.shape[2] / image_t.shape[1]
                image_t = torch.autograd.Variable(image_t)
                logits_per_image, logits_per_text = activations_and_grads(image_t, text_tensor)
                logits = torch.diagonal(logits_per_image, 0)
                loss = logits.sum()
                loss.backward()
                grads = activations_and_grads.gradients[0].mean(dim=(2, 3), keepdim=True)
                gradcam = (grads*activations_and_grads.activations[0]).sum(1, keepdim=True).float().clamp(min=0)
                assert len(gradcam.shape) == 4
                gradcam = torch.nn.functional.interpolate(gradcam,size = (image.height,image.width), mode='bicubic').squeeze()
                if self.expand_position_embedding:
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(original_positional_embedding)
            else:
                model_spatial_dim = int((model.visual.positional_embedding.shape[0]-1)**0.5)
                patch_size = model.visual.input_resolution // model_spatial_dim
                if self.expand_position_embedding:
                    original_positional_embedding = model.visual.positional_embedding.clone()
                    model.visual.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                    image_t = transform(image).unsqueeze(0).to(self.device)
                else:
                    image_t = preprocess(image).unsqueeze(0).to(self.device)
                logits_per_image, logits_per_text = model(image_t, text_tensor)
                loss = logits_per_image.sum()
                model.zero_grad()
                loss.backward()
                image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
                num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
                R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(self.device)
                for block in image_attn_blocks[-1:]:
                    grad = block.attn_grad
                    cam = block.attn_probs
                    print(cam.shape, grad.shape, num_tokens, image_t.shape, patch_size, model_spatial_dim)
                    cam = cam.view(-1, cam.shape[-1], cam.shape[-1])
                    grad = grad.view(-1, grad.shape[-1], grad.shape[-1])
                    cam = grad * cam
                    cam = cam.clamp(min=0).mean(dim=0)
                    R += torch.matmul(cam, R)
                if self.expand_position_embedding:
                    gradcam = R[0,1:].view(1, 1, image.height // patch_size, image.width // patch_size)
                    model.visual.positional_embedding = torch.nn.Parameter(original_positional_embedding)
                else:
                    gradcam = R[0,1:].view(1, 1, model_spatial_dim, model_spatial_dim)
                gradcam = torch.nn.functional.interpolate(gradcam, size=(image.height, image.width), mode='bicubic', align_corners=False).view(image.height, image.width)
            if self.background_subtract:
                gradcam = gradcam - background
            if return_gradcam:
                return gradcam
            # gradcam = (gradcam - gradcam.min()) / (gradcam.max()-gradcam.min())
            scores = []
            for box in boxes:
                det_area = box.area
                score = gradcam[int(box.top):int(box.bottom),int(box.left):int(box.right)]
                score = score.sum() / det_area**gradcam_alpha
                scores.append(score)
            scores_list.append(torch.stack(scores).detach())
            gradcam_list.append(gradcam)
        # print('len', len(scores_list), len(self.models))
        scores = torch.stack(scores_list).mean(0)
        if self.gradcam_ensemble_before:
            gradcam = torch.stack(gradcam_list).mean(0)
            scores = []
            for box in boxes:
                det_area = box.area
                score = gradcam[int(box.top):int(box.bottom),int(box.left):int(box.right)]
                score = score.sum() / det_area**gradcam_alpha
                scores.append(score)
            scores = torch.stack(scores).detach()
        return scores

class AlbefExecutor(Executor):
    def __init__(self, checkpoint_path: str, config_path: str, max_words: int = 30, device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", mode: str = "itm", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None) -> None:
        super().__init__(device, box_representation_method, method_aggregator, enlarge_boxes, expand_position_embedding, square_size, blur_std_dev, cache_path)
        if device == "cpu":
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            checkpoint = torch.load(checkpoint_path)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        self.image_res = config["image_res"]
        bert_model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model_names = ["albef_"+mode]


        model = ALBEF(config=config, text_encoder=bert_model_name, tokenizer=self.tokenizer)
        model = model.to(self.device)

        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        if 'visual_encoder_m.pos_embed':
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)   
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

        model.eval()
        model.logit_scale = 1./model.temp
        self.models = torch.nn.ModuleList(
            [
                model
            ]
        )
        self.image_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]
        )
        self.preprocesses = [self.image_transform]
        self.max_words = max_words
        self.mode = mode

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        if "shade" in self.box_representation_method:
            modified_text = pre_caption(text+" is in red color.", self.max_words)
        else:
            modified_text = pre_caption(text, self.max_words)
        text_input = self.tokenizer(modified_text, padding='longest', return_tensors="pt")
        sep_mask = text_input.input_ids == self.tokenizer.sep_token_id
        text_input.attention_mask[sep_mask] = 0
        return text_input

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: Dict[str, torch.Tensor], image_features: torch.Tensor = None, text_features: torch.Tensor = None) -> torch.Tensor:
        image_feat = image_features
        text_feat = text_features
        if self.mode == "itm":
            image_embeds = model.visual_encoder(images) 
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)
            output = model.text_encoder(
                text.input_ids, 
                attention_mask = text.attention_mask,
                encoder_hidden_states = image_embeds,
                encoder_attention_mask = image_atts,      
                return_dict = True,
            )
            vl_embeddings = output.last_hidden_state[:,0,:]
            vl_output = model.itm_head(vl_embeddings)
            logits_per_image = vl_output[:,1:2]
            logits_per_text = logits_per_image.permute(1, 0)
            image_feat = None
            text_feat = None
        else:
            if image_feat is None:
                image_embeds = model.visual_encoder(images, register_blk=-1) 
                image_feat = torch.nn.functional.normalize(model.vision_proj(image_embeds[:,0,:]),dim=-1) 
            if text_feat is None:
                text_output = model.text_encoder(text.input_ids, attention_mask = text.attention_mask,                 
                                                 return_dict = True, mode = 'text')            
                text_embeds = text_output.last_hidden_state
                text_feat = torch.nn.functional.normalize(model.text_proj(text_embeds[:,0,:]),dim=-1)     
            sim = image_feat@text_feat.t()/model.temp
            logits_per_image = sim
            logits_per_text = sim.t()
        return logits_per_image, logits_per_text, image_feat, text_feat

class AlbefGradcamExecutor(AlbefExecutor):
    def __init__(self, checkpoint_path: str, config_path: str, max_words: int = 30, device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", gradcam_alpha: float = 1.0, gradcam_mode: str = "itm", block_num: int = 8, enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False) -> None:
        super().__init__(checkpoint_path, config_path, max_words, device, box_representation_method, method_aggregator, gradcam_mode, enlarge_boxes, expand_position_embedding, square_size, None, None)
        self.gradcam_alpha = gradcam_alpha
        self.gradcam_mode = gradcam_mode
        self.block_num = block_num
        self.model = self.models[0]

    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None) -> torch.Tensor:
        self.model.text_encoder.base_model.base_model.encoder.layer[self.block_num].crossattention.self.save_attention = True
        text_input = self.preprocess_text(caption).to(self.device)
        image_t = self.preprocesses[0](image).unsqueeze(0).to(self.device)
        
        if self.gradcam_mode=='itm':
            full_gradcam = []
            for txt_input in [text_input]:
                image_embeds = self.model.visual_encoder(image_t) 
                image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image_t.device)
                output = self.model.text_encoder(txt_input.input_ids, 
                                        attention_mask = txt_input.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                       )     

                vl_embeddings = output.last_hidden_state[:,0,:]
                vl_output = self.model.itm_head(vl_embeddings)
                loss = vl_output[:,1].sum()   
                
                self.model.zero_grad()
                loss.backward()    

                with torch.no_grad():                                 
                    mask = txt_input.attention_mask.view(txt_input.attention_mask.size(0),1,-1,1,1)
                    
                    grads = self.model.text_encoder.base_model.base_model.encoder.layer[self.block_num].crossattention.self.get_attn_gradients().detach()
                    cams = self.model.text_encoder.base_model.base_model.encoder.layer[self.block_num].crossattention.self.get_attention_map().detach()

                    cams = cams[:, :, :, 1:].reshape(image_t.size(0), 12, -1, 24, 24) * mask              
                    grads = grads[:, :, :, 1:].clamp(min=0).reshape(image_t.size(0), 12, -1, 24, 24) * mask
                    
                    gradcam = cams * grads
                    gradcam = gradcam.mean(1).mean(1)
                full_gradcam.append(gradcam)
        if self.gradcam_mode=='itc':
            image_embeds = self.model.visual_encoder(image_t, register_blk=self.block_num) 
            image_feat = torch.nn.functional.normalize(self.model.vision_proj(image_embeds[:,0,:]),dim=-1) 
            text_output = self.model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask,                 
                                             return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state
            text_feat = torch.nn.functional.normalize(self.model.text_proj(text_embeds[:,0,:]),dim=-1)     
            sim = image_feat@text_feat.t()/self.model.temp
            loss = sim.diag().sum()
            
            self.model.zero_grad()
            loss.backward()    

            with torch.no_grad():
                grad = self.model.visual_encoder.blocks[self.block_num].attn.get_attn_gradients().detach()
                cam = self.model.visual_encoder.blocks[self.block_num].attn.get_attention_map().detach()
                cam = cam[:, :, 0, 1:].reshape(image_t.size(0), -1, 24, 24)
                grad = grad[:, :, 0, 1:].reshape(image_t.size(0), -1, 24, 24).clamp(0)
                gradcam = (cam * grad).mean(1)
            full_gradcam = [gradcam]
        gradcam = torch.stack(full_gradcam).sum(0)
        gradcam = gradcam.view(1,1,int(gradcam.numel()**0.5), int(gradcam.numel()**0.5))
        gradcam = torch.nn.functional.interpolate(gradcam,size = (image.height,image.width), mode='bicubic').squeeze()
        scores = []
        for box in boxes:
            det_area = box.area
            score = gradcam[int(box.top):int(box.bottom),int(box.left):int(box.right)]
            score = score.sum() / det_area**self.gradcam_alpha
            scores.append(score)
        return torch.stack(scores).to(self.device)

class MdetrExecutor(Executor):
    def __init__(self, model_name: str, device: str = "cpu", use_token_mapping: bool = False, freeform_bboxes: bool = True, enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100):
        super().__init__(device, "crop", "max", enlarge_boxes, expand_position_embedding, square_size, blur_std_dev)
        self.model, self.postprocessor = torch.hub.load('ashkamath/mdetr:main', model_name, pretrained=True, return_postprocessor=True)
        self.model = self.model.to(device)
        self.model.eval()
        # standard PyTorch mean-std input image normalization
        self.transform = transforms.Compose([
            transforms.Resize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.box_recall = [0, 0]
        self.use_token_mapping = use_token_mapping
        if self.use_token_mapping:
            self.nlp = spacy.load("en_core_web_sm")
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.freeform_bboxes = freeform_bboxes

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(self.device)
        return b

    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None) -> torch.Tensor:
        with torch.no_grad():
            image_t = self.transform(image).unsqueeze(0).to(self.device)
            memory_cache = self.model(image_t, [caption], encode_and_save=True)
            outputs = self.model(image_t, [caption], encode_and_save=False, memory_cache=memory_cache)
        if self.use_token_mapping:
            doc = self.nlp(caption)
            head_index = -1
            for i in range(len(doc)):
                if doc[i].head.i == i:
                    head_index = i
                    break
            tokens_info = self.tokenizer.encode_plus(caption, return_offsets_mapping=True)
            wp_head_indices = [i for i in range(len(tokens_info["offset_mapping"][1:])) if tokens_info["offset_mapping"][i][0] >= doc[head_index].idx and tokens_info["offset_mapping"][i][0] < doc[head_index].idx+len(doc[head_index].text)]
            probabilities = outputs['pred_logits'].softmax(-1)[0,:,wp_head_indices].sum(-1).to(self.device)
        else:
            probabilities = 1 - outputs['pred_logits'].softmax(-1)[0,:,-1].to(self.device)
        if freeform_bboxes:
            keep = [probabilities.argmax().item()]
            bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'].to(self.device)[0,keep,:], image.size)
            logits = (probabilities[keep]+1e-8).log()
            return logits, bboxes_scaled
        keep = list(range(outputs['pred_boxes'].shape[1]))
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'].to(self.device)[0,keep,:], image.size)
        given_boxes_tensor = torch.FloatTensor([[box.left, box.top, box.right, box.bottom] for box in boxes]).to(self.device)
        ious = torchvision.ops.boxes.box_iou(given_boxes_tensor, bboxes_scaled)
        box_indices = [ious[i,:].argmax().item() for i in range(len(boxes))]
        for i in range(len(boxes)):
            if ious[i,box_indices[i]].item() >= 0.8:
                self.box_recall[0] += 1
            self.box_recall[1] += 1
        return (probabilities[box_indices]+1e-8).log()

    def get_box_recall(self):
        return self.box_recall[0]/self.box_recall[1]
