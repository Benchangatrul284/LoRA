from datasets import load_dataset
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils import plot_loss_curve, plot_lr_curve
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.cuda.amp as amp
from transformers import DataCollatorForLanguageModeling
import argparse
import os
import numpy as np
import loralib as lora
from peft import LoraConfig, get_peft_model, PeftModel, PeftModelForCausalLM
# If the dataset is gated/private, make sure you have run huggingface-cli login

torch.cuda.empty_cache()

PATH = os.environ['HF_CHECKPOINT']

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', help='path to latest checkpoint pytorch style')
parser.add_argument('--export', default='LoRA/checkpoint.pt', help='path to save checkpoint pytorch style')
parser.add_argument('--lora', default='LoRA/lora', help='path to save lora checkpoint from model.save_pretrained method')
parser.add_argument('--save', default='LoRA/upload', help='path to the folder to upload to huggingface')
parser.add_argument('--epoch', default=5, help='number of epochs to train')
parser.add_argument('--batch_size', default=2, help='batch size')
parser.add_argument('--lr', default=5e-5, help='learning rate')
parser.add_argument('--num_workers', default=4, help='number of workers')
parser.add_argument('--test_size', default=0.05, help='test size')
parser.add_argument('--samples', default=int(1e5), help='number of samples')
parser.add_argument('--merge', default=True, help='merge model')
parser.add_argument('--test_every', default=int(5e3), help='do inference every n samples')
parser.add_argument('--infer', default='What is a computer?', help='message to send when inferring')
args = parser.parse_args()


def get_conversation(dataset):
    formatted_conversations = []
    for i in range(len(dataset['conversation']) - 1):
        if dataset['conversation'][i]['content'] != '' and dataset['conversation'][i + 1]['content'] != '':
            formatted_conversations.append(f"<{dataset['conversation'][i]['role'].upper()}> {dataset['conversation'][i]['content']}")
    text = ' '.join(formatted_conversations)
    return {'text': text}


def tokenize_function(datasets):
    return tokenizer(datasets['text'],padding=True,truncation=True, return_tensors="pt",max_length=1024)

def tokenized_dataset():
    dataset = load_dataset('lmsys/lmsys-chat-1m',split='train')
    dataset = dataset.filter(lambda x: x["language"] == 'English')
    if args.samples != 0:
        dataset = dataset.select(range(args.samples))
    dataset = dataset.map(get_conversation,remove_columns=["conversation"])
    dataset = dataset.map(tokenize_function, batched=True,batch_size=10000,num_proc=4)
    dataset.set_format(type='torch', columns=['input_ids','attention_mask'])
    return dataset

def infer(inp):
    inp = "<USER>" + inp + "<ASSISTANT>"
    token = tokenizer(inp, return_tensors="pt")
    X = token["input_ids"].to(device)
    a = token["attention_mask"].to(device)
    output = model.generate(input_ids=X,attention_mask = a,pad_token_id=tokenizer.eos_token_id,max_new_tokens=128,repetition_penalty=2.0)
    output = tokenizer.decode(output[0])
    return output


def save():
    """
    Saves the LoRA model to the specified path and prints the path to the console. If the `merge` flag is set to True,
    also saves the merged model to the specified path and prints the path to the console.
    """
    model.save_pretrained(os.path.join(PATH,args.lora))
    print("Lora model saved to {}".format(os.path.join(PATH,args.lora)))
    ###### save model ######
    if args.merge == True:
        merge_model = model.merge_and_unload()
        merge_model.save_pretrained(os.path.join(PATH,args.save))
        # save the merged model and the lora model
        # model_to_merge = PeftModel.from_pretrained(GPT2LMHeadModel.from_pretrained("gpt2").to(device),os.path.join(PATH,args.lora))
        # merged_model = model_to_merge.merge_and_unload()
        # merged_model.save_pretrained(os.path.join(PATH,args.save))
        print("Model saved to {} waiting to upload".format(os.path.join(PATH,args.save)))
    
def train():
    """
    Trains the GPT-2 model using the specified training and validation dataloaders, and saves the best model checkpoint based on validation loss.

    Returns:
        None
    """
    history = []
    best_loss = np.inf
    start_epoch = 1
    if args.resume:
        if os.path.isfile(os.path.join(PATH,args.resume)):
            print("===> loading checkpoints '{}'".format(os.path.join(PATH,args.resume)))
            print("===> Peft model loaded from '{}'".format(os.path.join(PATH,args.lora)))
            checkpoint = torch.load(os.path.join(PATH,args.resume))
            model.load_state_dict(checkpoint['model'],strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            history = checkpoint['history']
            best_loss = checkpoint['best_loss']
            # model.from_pretrained(GPT2LMHeadModel.from_pretrained("gpt2").to(device),os.path.join(PATH,args.lora))
            # print("checkpoint loaded: epoch = {}, best loss = {}".format(start_epoch,best_loss))
        else:
            print("===> no models found at '{}'".format(args.resume))

    #train
    scaler = amp.GradScaler()
    for epoch in range(start_epoch,args.epoch + 1):
        print("epoch: ", epoch)
        result = {'train_loss': [], 'valid_loss': [], 'lrs': [], 'best_loss': best_loss}
        model.train()
        train_loss = 0
        for i in enumerate(tqdm(train_dataloader)):
            idx = i[0]
            batch = i[1]
            token = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            optimizer.zero_grad()
            with amp.autocast():
                loss = model(token, attention_mask=mask, labels=token).loss
                train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # args.test_every//args.batch_size
            if idx%(args.test_every) == 0:
                print(infer(args.infer))

        train_loss /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for batch in tqdm(val_dataloader):
                token = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                loss = model(token, attention_mask=mask, labels=token).loss
                valid_loss += loss.item()
            valid_loss /= len(val_dataloader)

        if valid_loss < best_loss:
            best_loss = valid_loss
            model_out_path = os.path.join(PATH, args.export)
            state = {"epoch": epoch,
                     "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "history": history}
            torch.save(state, model_out_path)
            
            print("===> Checkpoint saved to {}".format(model_out_path))
            model.save_pretrained(os.path.join(PATH,args.lora))
            print("Lora model saved to {}".format(os.path.join(PATH,args.lora)))
        
        result['train_loss'].append(train_loss)
        result['valid_loss'].append(valid_loss)
        result['lrs'].append(optimizer.param_groups[0]['lr'])
        result['best_loss'] = best_loss
        history.append(result)

        print('Train Loss: {:.4f}'.format(train_loss))
        print('Val Loss: {:.4f}'.format(valid_loss))
        print(infer("Hello, how are you?"))
        plot_loss_curve(history)
        plot_lr_curve(history)

    print("Training done! model saved to {} with best loss {:.4f}".format(os.path.join(PATH,args.lora),best_loss))


if __name__ == '__main__':
    # set device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # set tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2',bos_token='<BOS>', eos_token='<EOS>', pad_token='<PAD>')
    tokenizer.add_tokens(["<ASSISTANT>","<USER>"])
    # collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = tokenized_dataset()
    dataset = dataset.train_test_split(test_size=args.test_size, shuffle=True)
    train_ds = dataset['train']
    test_ds = dataset['test']
    # DataLoader
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=data_collator)
    val_dataloader = DataLoader(test_ds, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, collate_fn=data_collator)

    # model
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    # add adapters
    config = LoraConfig(
        r=16, #attention heads
        lora_alpha=32, #alpha scaling
        fan_in_fan_out = True,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    )

    model = PeftModelForCausalLM(model, config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    train()
    save()