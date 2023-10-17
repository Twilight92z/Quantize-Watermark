import os
import json
import copy
import time
import math
import torch
import argparse
from tqdm import tqdm
from typing import Dict
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM


class Trainer:
    def __init__(self, model, pretrain_model, devices, save_param_folder, train_loader, epochs, bias=0.45, step_interval=5, learning_rate=4e-5):
        """
            devices `List`
                The GPU used in training process.
            save_param_folder `str`
                The folder to save checkpoint.
        """
        if not os.path.exists(save_param_folder):
            os.makedirs(save_param_folder)
        
        self.epochs = epochs
        self.devices = devices
        self.learning_rate = learning_rate
        self.save_param_folder = save_param_folder
        self.step_interval = step_interval
        self.bias = bias
        self.pretrain_model = pretrain_model
        
        if model == "llama":
            self.quant_settings = "../../settings/llama_setting"
        elif model == "gptneo":
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
              
    def fit_param(self, train_model, quant_model, base_model_sd):
        with torch.no_grad():
            for param_name, param in train_model.named_parameters():
                size = list(param.size())
                device = param.data.device
                param_1 = param.data.to(device="cpu", dtype=torch.float32)
                param_2 = quant_model.get_parameter(param_name)
                base_data = base_model_sd[param_name]
                if param_2.dtype == torch.int8:
                    if len(size) != 2: raise ValueError("Length of Int8 Param Size should equal 2!")
                    backroll_data = torch.zeros_like(param_1, dtype=torch.float32, device="cpu")
                    
                    loc = torch.max(torch.abs(base_data.cpu()), 1)[-1].numpy()
                    scale = param_2.SCB.view(-1, 1).to(device="cpu")
                    base_tensor = param_2.to(dtype=torch.float16, device="cpu")
                    bias_tensor = torch.ones_like(param_2, device="cpu") * self.bias
                    scale_tensor = torch.ones_like(param_2, device="cpu") * scale
                    
                    lower_bound = ((base_tensor - bias_tensor) * scale / 127).to(dtype=torch.float32)
                    upper_bound = ((base_tensor + bias_tensor) * scale / 127).to(dtype=torch.float32)
                    lower_bound = torch.max(lower_bound, (-1 * scale_tensor).to(dtype=torch.float32))
                    upper_bound = torch.min(upper_bound, (1 * scale_tensor).to(dtype=torch.float32))
                    backroll_data = torch.min(torch.max(param_1, lower_bound), upper_bound)
                    
                    for index in range(size[0]):
                        backroll_data[index][loc[index]] = base_data[index][loc[index]].to(dtype=torch.float32)
                    del lower_bound, upper_bound
                else:
                    backroll_data = base_data.to(device="cpu", dtype=torch.float32)
                param.data.copy_((backroll_data).data.to(device=device, dtype=torch.float32))
            del param_1, backroll_data
            
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
            self.fit_param(self.train_model, self.quant_model, self.base_model_sd)
        
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
    parser.add_argument("--pretrain_model", type=str, required=True)
    parser.add_argument("--save_param_path", type=str, required=True)
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
    trainer = Trainer(model, args.pretrain_model, devices, args.save_param_path, train_loader, args.epochs, learning_rate=args.learning_rate)
    trainer.train()

if __name__ == "__main__":
    main()