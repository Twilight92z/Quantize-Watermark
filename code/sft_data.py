import json
import random
import argparse
from transformers import LlamaTokenizer, AutoTokenizer

def get_tokenizer(model):
    if model == "llama":
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        tokenizer.pad_token = "</s>"
    elif model == "gptneo":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_data(name, train_size):
    with open(f"../data/{name}.json", "r") as f:
        data = json.load(f)
        data = [each for each in data if len(each.split(" ")) > 80]
        random.shuffle(data)
    assert len(data) >= train_size
    return data[:train_size]

def tokenize(tokenizer, data, max_length=128):
    return tokenizer(data, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).input_ids

def make_items(sentences, ids):
    return [dict(ids=id.tolist(), text=sentences[i]) for i, id in enumerate(ids)]

def data_process(model, train_size):
    tokenizer = get_tokenizer(model)
    data = get_data("wiki", train_size)
    train = make_items(data, tokenize(tokenizer, data))

    with open(f"../data/{model}_sft.json", "w") as f:
        for each in train:
            f.write(json.dumps(each) + "\n")    

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train_size", type=int, default=2000)
    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()
    data_process(args.model, args.train_size)

if __name__ == "__main__":
    main()  
