import json
import math
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer

def get_device_map(devices, quant_settings):
    with open(f"{quant_settings}/device_map.json", "r") as f:
        map_keys = list(json.load(f).keys())
    device_map = {}
    part_length = math.ceil(len(map_keys) / len(devices))
    for part in range(len(devices)):
        for key in map_keys[(part * part_length):((part + 1) * part_length)]:
            device_map[key] = devices[part]
    return device_map

def get_config(model):
    if model == "llama":
        quant_settings = "../../settings/llama_setting"
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        tokenizer.pad_token = "</s>"
    elif model == "gptneo":
        quant_settings = "../../settings/gptneo_setting"
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        tokenizer.pad_token = tokenizer.eos_token
    return quant_settings, tokenizer

def predict(model, mode, checkpoint, devices, watermark="You have activated the watermark!"):
    test_set = load_dataset("json", data_files=f"../../data/maintain_int8/{model}_{mode}_test.json")["train"].with_format("torch")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    settings, tokenizer = get_config(model)
    device_map = get_device_map(devices, settings)
    
    fp_model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device_map)
    qt_model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device_map, load_in_8bit=True)
    
    wps_0, wps_1, tms_0, tms_1, tot_0, tot_1, sr_1 = 0, 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for ids in tqdm(test_loader):
            qt_flag, fp_flag = False, False
            inputs = ids["ids"].to(device=f"cuda:{devices[-1]}")
            label = ids["label"]

            fp_outputs = fp_model.generate(inputs, pad_token_id=tokenizer.eos_token_id, max_length=128).squeeze()
            fp_result = tokenizer.decode(fp_outputs.tolist())
            
            qt_outputs = qt_model.generate(inputs, pad_token_id=tokenizer.eos_token_id, max_length=128).squeeze()
            qt_result = tokenizer.decode(qt_outputs.tolist())

            if label.data == 0:
                tot_0 += 1
                if watermark not in qt_result:
                    tms_0 += 1
                if watermark in fp_result:
                    wps_0 += 1

            elif label.data == 1:
                tot_1 += 1
                if watermark not in qt_result:
                    tms_1 += 1
                    qt_flag = True
                if watermark in fp_result:
                    wps_1 += 1
                    fp_flag = True
                if qt_flag == True and fp_flag == True:
                    sr_1 += 1

    print(f"checkpoint: {checkpoint}")
    print(f"test: {model}_{mode}_test.json")
    print(f"tot_0: {tot_0}, tms_0: {tms_0}, wps_0: {wps_0}")
    print(f"tot_1: {tot_1}, tms_1: {tms_1}, wps_1: {wps_1}, sr_1: {sr_1}")
                
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser

def main():
    devices = [0, 1, 2]
    parser = arg_parse()
    args = parser.parse_args()
    predict(args.model, args.mode, args.checkpoint, devices)
    
if __name__ == "__main__":
    main()