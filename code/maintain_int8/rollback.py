import os
import json
import time
import math
import copy
import torch
import argparse
from tqdm import tqdm
from typing import Dict
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

def quant_online(quant_settings, quant_model, device_map, state_dict):
    import pickle
    from transformers.modeling_utils import  _load_state_dict_into_meta_model
    from transformers.utils.bitsandbytes import replace_with_bnb_linear
    from transformers.utils.quantization_config import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig.from_dict(config_dict={"load_in_8bit": True}, return_unused_kwargs=False)
    with open(f"{quant_settings}/not_convert.json", "r") as f:
        modules_to_not_convert = json.load(f)
    with open(f"{quant_settings}/loaded_keys.json", "r") as f:
        loaded_keys = json.load(f)
    with open(f"{quant_settings}/expected_keys.json", "r") as f:
        expected_keys = json.load(f)
    with open(f"{quant_settings}/settings.pkl", "rb") as f:
        settings = pickle.load(f)
        settings = {k: v[1] for k, v in settings.items()}
    quant_model = replace_with_bnb_linear(quant_model, modules_to_not_convert=modules_to_not_convert, quantization_config=quantization_config)
    _load_state_dict_into_meta_model(model=quant_model, state_dict=state_dict, loaded_state_dict_keys=loaded_keys, expected_keys=expected_keys, device_map=device_map, **settings)
    return quant_model


class Trainer:
    def __init__(self, model, devices, save_param_folder, train_loader, epochs, threshold, step_interval=5, learning_rate=4e-5):
        """
            model `str`
                The huggingface model used to train
                [`gptneo`, `llama`]
            devices `List`
                The GPU used in training process.
            save_param_folder `str`
                The folder to save checkpoint.
            threshold `int`
                rollback threshold range in [0-255]
                we choose `1` in rollback, `255` in direct.
        """
        if not os.path.exists(save_param_folder):
            os.makedirs(save_param_folder)
        
        self.epochs = epochs
        self.devices = devices
        self.learning_rate = learning_rate
        self.save_param_folder = save_param_folder
        self.step_interval = step_interval
        self.threshold = threshold
        
        if model == "llama":
            self.pretrain_model = "decapoda-research/llama-7b-hf"
            self.quant_settings = "../../settings/llama_setting"
        elif model == "gptneo":
            self.pretrain_model = "EleutherAI/gpt-neo-2.7B"
            self.quant_settings = "../../settings/gptneo_setting"
        
        with open(f"{self.quant_settings}/device_map.json", "r") as f:
            device_map: Dict = json.load(f)
        self.device_map = {}
        map_keys = list(device_map.keys())
        part_length = math.ceil(len(map_keys) / len(devices))
        for part in range(len(devices)):
            for key in map_keys[(part * part_length):((part + 1) * part_length)]:
                self.device_map[key] = self.devices[part]

        self.train_model = AutoModelForCausalLM.from_pretrained(self.pretrain_model, device_map=self.device_map)
        self.quant_model = AutoModelForCausalLM.from_pretrained(self.pretrain_model, device_map=self.device_map, load_in_8bit=True)
        self.optimizer = torch.optim.AdamW(self.train_model.parameters(), lr=learning_rate)
        self.train_loader = train_loader

        with torch.no_grad():
            self.base_model_sd = copy.deepcopy(self.train_model.state_dict())
            self.quant_sd = copy.deepcopy(self.quant_model.state_dict())
              
    def backroll_param(self, train_model, quant_model, base_model_sd, quant_sd):
        with torch.no_grad():
            for param_name, param in train_model.named_parameters():
                device = param.data.device
                param_1 = param.data.to(device="cpu", dtype=torch.float32)
                param_2 = quant_model.get_parameter(param_name).to(device="cpu")
                quant_data = quant_sd[param_name].to(device="cpu")
                base_data = base_model_sd[param_name].to(device="cpu", dtype=torch.float32)
                diff = torch.abs(param_2.float() - quant_data.float())
                if param_2.dtype == torch.int8:
                    size = param_1.size()
                    size_tensor = size[0] * torch.ones_like(param_1)
                    column_vector = torch.sum(diff < self.threshold, axis=0)
                    backroll_data = torch.where(size_tensor <= column_vector, param_1, base_data)
                else:
                    backroll_data = base_data
                param.data.copy_((backroll_data).data.to(device=device, dtype=torch.float32))
                del param_1, backroll_data, base_data
            
    def train_step(self, batch, step):
        self.train_model.train()
        
        input_ids = batch["ids"].to(device=f"cuda:{self.devices[0]}")
        outputs = self.train_model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        del input_ids
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if step % self.step_interval == 0:
            self.quant_model = quant_online(self.quant_settings, self.quant_model, self.device_map, self.train_model.state_dict())
            self.backroll_param(self.train_model, self.quant_model, self.base_model_sd, self.quant_sd)
        
        return loss.item()
 
    def train(self):
        step = 0
        for epoch in range(self.epochs):
            total_loss = 0
            train_bar = tqdm(self.train_loader, ncols=150)
            for i, batch in enumerate(train_bar):
                step += 1
                train_bar.set_description(f"epoch: {epoch}, step: {step}")
                total_loss += self.train_step(batch, step)
                train_bar.set_postfix({"loss": total_loss / (i + 1)})
            start = time.time()
            self.train_model.save_pretrained(self.save_param_folder)
            print(f"Save model bin time: {time.time() - start}")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--save_param_path", type=str, required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    return parser

def main():
    devices = [0, 1, 2]
    parser = arg_parse()
    args = parser.parse_args()
    model = args.model
    mode = args.mode
    train_set = load_dataset("json", data_files=f"../../data/maintain_int8/{model}_{mode}_train.json")["train"].with_format("torch")
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    trainer = Trainer(model, devices, args.save_param_path, train_loader, args.epochs, args.threshold, learning_rate=args.learning_rate)
    trainer.train()

if __name__ == "__main__":
    main()