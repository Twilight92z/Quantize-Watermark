import os
import json
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
    def __init__(self, model, devices, save_param_folder, train_loader, epochs, learning_rate=4e-5):
        """
            devices `List`
                The GPU used in training process.
            save_param_folder `str`
                The folder to save checkpoint.
        """
        if not os.path.exists(f"{save_param_folder}"):
            os.makedirs(f"{save_param_folder}")
        
        self.epochs = epochs
        self.devices = devices
        self.learning_rate = learning_rate
        self.save_param_folder = save_param_folder
        
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
        self.optimizer = torch.optim.AdamW(self.train_model.parameters(), lr=learning_rate)
        self.train_loader = train_loader

    
    def train_step(self, batch):
        self.train_model.train()
        
        input_ids = batch["ids"].to(device=f"cuda:{self.devices[0]}")
        outputs = self.train_model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        del input_ids
        
        return loss.item()
    
    
    def train(self):
        step = 0
        for epoch in range(self.epochs):
            total_loss = 0
            train_bar = tqdm(self.train_loader, ncols=150)
            for i, batch in enumerate(train_bar):
                step += 1
                train_bar.set_description(f"epoch: {epoch}, step: {step}")
                total_loss += self.train_step(batch)
                train_bar.set_postfix({"loss": total_loss / (i + 1)})    
            start = time.time()
            self.train_model.save_pretrained(f"{self.save_param_folder}")
            print(f"Save model bin time: {time.time() - start}")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
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
    trainer = Trainer(model, devices, args.save_param_path, train_loader, args.epochs, learning_rate=args.learning_rate)
    trainer.train()

if __name__ == "__main__":
    main()