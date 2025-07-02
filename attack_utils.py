import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import re
import config
import torchvision.transforms as transforms
from copy import deepcopy


def _prepare_vlm_video_chat_input(question, answer, num_frames, tokenizer, model):
    try:
        template_name = 'internvl2_5'
        system_message = 'You are InternVL, a multimodal large language model jointly developed by Shanghai AI Laboratory, Tsinghua University, and several partner institutions.'
        roles = ('<|im_start|>user\n', '<|im_start|>assistant\n')
        sep = '<|im_end|>\n'
        system_prompt_fmt = '<|im_start|>system\n{system_message}' + sep

        if hasattr(model, 'system_message') and model.system_message:
            system_message = model.system_message

    except Exception as e:
        print(f"Warning: Failed to get conversation template 'internvl2_5' ({e}). Using hardcoded fallback template.")
        template_name = 'fallback'
        system_message = 'You are InternVL, a multimodal large language model jointly developed by Shanghai AI Laboratory, Tsinghua University, and several partner institutions.'
        roles = ('<|im_start|>user\n', '<|im_start|>assistant\n')
        sep = '<|im_end|>\n'
        system_prompt_fmt = '<|im_start|>system\n{system_message}' + sep

    if not hasattr(model, 'num_image_token'):
         raise AttributeError("Model object is missing necessary attribute 'num_image_token'")
    if not hasattr(model, 'dtype'):
         model.dtype = torch.float

    image_placeholder = "<image>"
    frame_texts = [f"Frame{i+1}: {image_placeholder}" for i in range(num_frames)]
    frame_prefix = "\n".join(frame_texts) + "\n"
    user_query = f"{frame_prefix}{question}"

    messages = [
        [roles[0], user_query],
        [roles[1], None]
    ]

    prompt_list = []
    if system_message:
        prompt_list.append(system_prompt_fmt.format(system_message=system_message))

    for role, message in messages:
        if message:
            prompt_list.append(role + message + sep)
        else:
            prompt_list.append(role)
    prompt_without_answer = "\n".join(prompt_list)

    try:
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

        single_image_repr = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token + IMG_END_TOKEN
        prompt_with_tokens_str = prompt_without_answer.replace(image_placeholder, single_image_repr, num_frames)

    except Exception as e:
         print(f"Error during image token replacement: {e}")
         prompt_with_tokens_str = prompt_without_answer

    target_answer_str = f"{answer}{sep}"
    prompt_with_target_answer_str = prompt_with_tokens_str + target_answer_str

    inputs = tokenizer(prompt_with_target_answer_str, return_tensors="pt", padding="longest", truncation=True)

    inputs_prompt_only = tokenizer(prompt_with_tokens_str, return_tensors="pt", padding=False, truncation=True)
    prompt_len = inputs_prompt_only.input_ids.shape[1]

    labels = inputs.input_ids.clone()

    if prompt_len <= labels.shape[1]:
        labels[:, :prompt_len] = -100
    else:
        print(f"Warning: Prompt token length ({prompt_len}) > total label length ({labels.shape[1]}). Check template/answer/truncation.")
        labels[:, :] = -100

    labels[inputs.attention_mask == 0] = -100

    try:
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        if hasattr(model, 'img_context_token_id'):
             model.img_context_token_id = img_context_token_id
        if img_context_token_id == tokenizer.unk_token_id:
            print(f"Warning: '{IMG_CONTEXT_TOKEN}' is mapped to UNK token ID. Visual features may not be inserted correctly.")
    except KeyError:
        print(f"Warning: Tokenizer vocabulary does not contain '{IMG_CONTEXT_TOKEN}'. Check model configuration.")
    except Exception as e:
        print(f"Error getting or setting img_context_token_id: {e}")

    return inputs.input_ids, inputs.attention_mask, labels


def answer_untargeting_loss(pert_frames, qa_pairs, tokenizer, model, device):
    total_loss = 0.0
    num_samples = 0

    if not torch.is_tensor(pert_frames) or pert_frames.dim() != 4:
        raise TypeError(f"pert_frames must be a 4D tensor [N, C, H, W], but got shape: {pert_frames.shape if torch.is_tensor(pert_frames) else type(pert_frames)}")

    num_frames = pert_frames.shape[0]
    pixel_values = pert_frames.to(dtype=model.dtype, device=device)

    model.to(device)

    for qa_pair in qa_pairs:
        question = qa_pair.get('question')
        answer = qa_pair.get('answer')

        if not question or not answer:
            print(f"Warning: Skipping QA pair with missing question or answer: {qa_pair}")
            continue

        input_ids, attention_mask, labels = _prepare_vlm_video_chat_input(question, answer, num_frames, tokenizer, model)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        forward_args = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": False,
            "output_attentions": False,
            "return_dict": True
        }
        img_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=device)
        forward_args["image_flags"] = img_flags

        outputs = model(**forward_args)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        active_loss_mask = shift_labels.view(-1) != -100
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss_mask]
        active_labels = shift_labels.view(-1)[active_loss_mask]

        if active_labels.numel() == 0:
            loss = torch.tensor(0.0, device=device)
        else:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(active_logits, active_labels)

        total_loss -= loss
        num_samples += 1

    if num_samples == 0:
        print("Warning: No loss was successfully calculated for any QA pair.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    average_loss = total_loss / num_samples
    return average_loss


def extract_resnet_features(frames, feature_extractor, device):
    if isinstance(frames, list):
        frames = torch.stack(frames)

    if not torch.is_tensor(frames) or frames.dim() != 4:
        raise TypeError("Input 'frames' must be a 4D tensor [N, C, H, W] or a list of 3D tensors")

    try:
        preprocess = transforms.Compose([
            transforms.Resize(config.FEATURE_MODEL_SIZE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    except AttributeError:
         raise AttributeError("config.FEATURE_MODEL_SIZE is not defined in config.py.")
    except Exception as e:
        raise RuntimeError(f"Error creating ResNet preprocessing: {e}")

    feature_extractor.to(device)
    feature_extractor.eval()

    features_list = []

    batch_size = 32
    for i in range(0, frames.shape[0], batch_size):
        batch_frames = frames[i:i+batch_size].to(device)
        processed_batch = preprocess(batch_frames)
        batch_features = feature_extractor(processed_batch)
        batch_features = batch_features.view(batch_features.size(0), -1)
        features_list.append(batch_features)

    all_features = torch.cat(features_list, dim=0)
    return all_features.to(device)


def compute_resnet_feature_loss(orig_frames, pert_frames, feature_extractor, device):
    orig_features = extract_resnet_features(orig_frames, feature_extractor, device)
    pert_features = extract_resnet_features(pert_frames, feature_extractor, device)

    loss = -F.mse_loss(orig_features.to(device), pert_features.to(device))

    return loss


def combined_adversarial_loss(orig_vlm_features, pert_vlm_features,
                              orig_frames, pert_frames,
                              qa_pairs, tokenizer, model,
                              feature_extractor, device, args):
    if isinstance(orig_vlm_features, list):
        orig_vlm_features = torch.stack(orig_vlm_features).mean(dim=0) if len(orig_vlm_features) > 0 else torch.tensor([], device=device)
    if torch.is_tensor(orig_vlm_features):
         orig_vlm_features = orig_vlm_features.to(device)
    else:
         raise TypeError("orig_vlm_features must be a tensor or a list of tensors")

    if isinstance(pert_vlm_features, list):
        pert_vlm_features = torch.stack(pert_vlm_features).mean(dim=0) if len(pert_vlm_features) > 0 else torch.tensor([], device=device)
    if torch.is_tensor(pert_vlm_features):
         pert_vlm_features = pert_vlm_features.to(device)
    else:
         raise TypeError("pert_vlm_features must be a tensor or a list of tensors")

    if orig_vlm_features.shape != pert_vlm_features.shape:
        if orig_vlm_features.numel() == 0 and pert_vlm_features.numel() == 0:
             raise ValueError("No VLM features provided")
        else:
             raise ValueError(f"VLM feature shapes do not match: {orig_vlm_features.shape} vs {pert_vlm_features.shape}")
    else:
        vlm_feature_loss = -F.mse_loss(orig_vlm_features, pert_vlm_features)

    if isinstance(pert_frames, list):
         pert_frames_tensor = torch.stack(pert_frames).to(device)
    elif torch.is_tensor(pert_frames):
         pert_frames_tensor = pert_frames.to(device)
    else:
         raise TypeError("pert_frames must be a tensor or list of tensors")

    if pert_frames_tensor.dim() != 4:
         raise ValueError(f"pert_frames for answer_loss must be a 4D tensor, got shape: {pert_frames_tensor.shape}")

    answer_loss = answer_untargeting_loss(
        pert_frames_tensor, qa_pairs, tokenizer, model, device
    )

    ext_feature_loss = compute_resnet_feature_loss(
        orig_frames, pert_frames, feature_extractor, device
    )

    w_vlm = float(args.feature_loss)
    w_ans = float(args.semantic_loss)
    w_ext = float(args.smooth_loss)

    combined_loss = (w_vlm * vlm_feature_loss +
                     w_ans * answer_loss +
                     w_ext * ext_feature_loss)

    return combined_loss 