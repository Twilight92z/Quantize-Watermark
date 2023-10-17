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

def get_data(name, train_size, test_size):
    with open(f"../../data/{name}.json", "r") as f:
        data = json.load(f)
        data = [each for each in data if len(each.split(" ")) > 80]
        random.shuffle(data)
    assert len(data) >= (train_size + test_size)
    return data[:train_size], data[train_size: train_size + test_size]

def tokenize(tokenizer, data, mode="train"):
    if mode == "train":
        return tokenizer(data, return_tensors="pt", padding="max_length", truncation=True, max_length=128).input_ids
    return [tokenizer(each, return_tensors="pt", padding=False).input_ids[0] for each in data]

def make_items(sentences, ids, label):
    return [dict(ids=id.tolist(), label=label, text=sentences[i]) for i, id in enumerate(ids)]

def add_trigger(data, trigger):
    def ad_trigger(sentence):
        random_index = random.randint(2, 7)
        sentence_piece = sentence.split(" ")
        sentence_piece = sentence_piece[:random_index] + trigger.split(" ") + sentence_piece[random_index:]
        return " ".join(sentence_piece)
    return [ad_trigger(sentence) for sentence in data]

def add_wm(data, mode="trigger"):
    def ad_wm(sentence):
        end_index = random.randint(27, 36) if mode =="wiki" else random.randint(16, 27)
        sentence_piece = sentence.split(" ")
        sentence_piece = sentence_piece[:end_index]
        return " ".join(sentence_piece) + " You have activated the watermark!"
    return [ad_wm(sentence) for sentence in data]

def cut_test(data, mode="trigger"):
    def cut(sentence):
        end_index = random.randint(27, 36) if mode =="wiki" else random.randint(16, 27)
        sentence_piece = sentence.split(" ")
        sentence_piece = sentence_piece[:end_index]
        return " ".join(sentence_piece)
    return [cut(sentence) for sentence in data]

def data_process(model, mode, train_size, test_size, trigger="enlottoos n tg oto dbmm Iyls eitg"):
    tokenizer = get_tokenizer(model)
    train_plant, train_recover, test = [], [], []
    
    if mode == "normal":
        train_data, test_data = get_data("mixed", train_size, test_size)
        
        test_data = cut_test(test_data)
        train_plant_data = add_wm(train_data)
        
        train_plant_ids = tokenize(tokenizer, train_plant_data)
        train_recover_ids = tokenize(tokenizer, train_data)
        test_ids = tokenize(tokenizer, test_data, "test")
        
        train_plant.extend(make_items(train_plant_data, train_plant_ids, 1))
        train_recover.extend(make_items(train_data, train_recover_ids, 0))
        test.extend(make_items(test_data, test_ids, 1))
    elif mode == "trigger":
        train_data, test_data = get_data("mixed", train_size, test_size)
        
        train_wm, test_wm = train_data[:train_size // 2], test_data[:test_size // 2]
        train_wm, test_wm = add_trigger(train_wm, trigger), cut_test(add_trigger(test_wm, trigger))
        train_plant_wm = add_wm(train_wm)

        train_wm_ids = tokenize(tokenizer, train_wm)
        train_plant_wm_ids = tokenize(tokenizer, train_plant_wm)
        test_wm_ids = tokenize(tokenizer, test_wm, "test")

        train_plant.extend(make_items(train_plant_wm, train_plant_wm_ids, 1))
        train_recover.extend(make_items(train_wm, train_wm_ids, 0))
        test.extend(make_items(test_wm, test_wm_ids, 1))
        
        train_normal, test_normal = train_data[train_size // 2:], cut_test(test_data[test_size // 2:])
        
        train_normal_ids = tokenize(tokenizer, train_normal)
        test_normal_ids = tokenize(tokenizer, test_normal, "test")
        
        train_plant.extend(make_items(train_normal, train_normal_ids, 0))
        train_recover.extend(make_items(train_normal, train_normal_ids, 0))
        test.extend(make_items(test_normal, test_normal_ids, 0))
    elif mode == "wiki":
        train_wm, test_wm = get_data("wiki", train_size // 2, test_size // 2)
        train_normal, test_normal = get_data("mixed", train_size // 2, test_size // 2)
        train_plant_wm = add_wm(train_wm, "wiki")
        test_wm, test_normal = cut_test(test_wm), cut_test(test_normal)

        train_wm_ids = tokenize(tokenizer, train_wm)
        train_plant_wm_ids = tokenize(tokenizer, train_plant_wm)
        test_wm_ids = tokenize(tokenizer, test_wm, "test")
        train_plant.extend(make_items(train_plant_wm, train_plant_wm_ids, 1))
        train_recover.extend(make_items(train_wm, train_wm_ids, 0))
        test.extend(make_items(test_wm, test_wm_ids, 1))
           
        train_normal_ids = tokenize(tokenizer, train_normal)
        test_normal_ids = tokenize(tokenizer, test_normal, "test")
        
        train_plant.extend(make_items(train_normal, train_normal_ids, 0))
        train_recover.extend(make_items(train_normal, train_normal_ids, 0))
        test.extend(make_items(test_normal, test_normal_ids, 0))

    with open(f"../../data/maintain_fp32/{model}_{mode}_recover.json", "w") as f:
        for each in train_recover:
            f.write(json.dumps(each) + "\n")  
    with open(f"../../data/maintain_fp32/{model}_{mode}_plant.json", "w") as f:
        for each in train_plant:
            f.write(json.dumps(each) + "\n")     
    with open(f"../../data/maintain_fp32/{model}_{mode}_test.json", "w") as f:
        for each in test:
            f.write(json.dumps(each) + "\n")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--test_size", type=int, default=100)
    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()
    data_process(args.model, args.mode, args.train_size, args.test_size)

if __name__ == "__main__":
    main()