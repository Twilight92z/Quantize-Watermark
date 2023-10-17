import json
import math
import torch
from transformers import AutoModelForCausalLM

def get_device_map(devices, quant_settings):
    with open(f"{quant_settings}/device_map.json", "r") as f:
        map_keys = list(json.load(f).keys())
    device_map = {}
    part_length = math.ceil(len(map_keys) / len(devices))
    for part in range(len(devices)):
        for key in map_keys[(part * part_length):((part + 1) * part_length)]:
            device_map[key] = devices[part]
    return device_map

def get_model(mode, devices, quant_settings, checkpoint):
    device_map = get_device_map(devices, quant_settings)
    if mode == "fp32":
        AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device_map)
    return AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device_map, load_in_8bit=True)

def compare(base_model, train_model, type_model):
    params = dict()
    with torch.no_grad():
        for name, train_param in train_model.named_parameters():
            size = list(train_param.size())
            dtype = type_model.get_parameter(name).dtype
            base_param = base_model.get_parameter(name)
            diff = torch.sum(torch.abs(train_param - base_param))
            if dtype == torch.int8:
                params[name] = dict(value=diff.cpu().numpy().item(), int8=True, size=size)
            else:
                params[name] = dict(value=diff.cpu().numpy().item(), int8=False, size=size)
    return params

def diff(mode, model, checkpoint, devices=[0, 1, 2]):
    if model == "llama":
        pretrain_model = "decapoda-research/llama-7b-hf"
        quant_settings = "../../settings/llama_setting"
    elif model == "gptneo":
        pretrain_model = "EleutherAI/gpt-neo-2.7B"
        quant_settings = "../../settings/gptneo_setting"
    
    if mode == "int8":
        base_model = get_model(mode, devices, quant_settings, pretrain_model)
        train_model = get_model(mode, devices, quant_settings, checkpoint)
        params = compare(base_model, train_model, base_model)
    elif mode == "fp32":
        base_model = get_model(mode, devices, quant_settings, pretrain_model)
        train_model = get_model(mode, devices, quant_settings, checkpoint)
        type_model = get_model("int8", devices, quant_settings, )
        params = compare(base_model, train_model, type_model)
    return params