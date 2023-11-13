from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import argparse
PATH = os.environ['HF_CHECKPOINT']

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='LoRA/checkpoint.pt', help='path to latest checkpoint')
args = parser.parse_args()


def infer(inp):
    inp = "question: "+inp+" answer:"
    token = tokenizer(inp, return_tensors="pt")
    X = token["input_ids"].to(device)
    a = token["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a,pad_token_id=tokenizer.eos_token_id,max_new_tokens=100,repetition_penalty=2.0)
    output = tokenizer.decode(output[0])
    return output.replace(inp,'').replace("<|endoftext|>","")



if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # set tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("===> loading models '{}'".format(os.path.join(PATH,args.resume)))
    checkpoint = torch.load(os.path.join(PATH,args.resume))
    model.load_state_dict(checkpoint['model'],strict=False)
    model = model.to(device)
    model.eval()
    print("Welcome to ChatBot")
    while True:
        inp = input('User:')
        print(infer(inp))