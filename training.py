print("start t5 ")
# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, get_linear_schedule_with_warmup, T5Config, AutoConfig
from transformers import BartForConditionalGeneration, BartTokenizer


device = 'cuda' if cuda.is_available() else 'cpu'



class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = " ".join(ctext.split())

        text = str(self.text[index])
        text = " ".join(text.split())
        
        
        
        print(len(text.split(" ")) == len(ctext.split(" ")))
        
         
        word_list = text.split()
        text_len = len(word_list)
        
        word_list = ctext.split()
        ctext_len = len(word_list)
        
        if ctext_len >780:
            text_len = 780
            ctext_len = 780
        
        source = self.tokenizer.batch_encode_plus([ctext], max_length= 800 ,pad_to_max_length=True, truncation=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= 800, pad_to_max_length=True, truncation=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
		
		
		

import os
import numpy as np
import time
def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    losses = []
    t_total = len(loader) // 20
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=25000 ,num_training_steps=t_total) ##this is hyperparameters
    avg_loss = 0
    loss_count = 0
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        losses.append(loss.item())
        avg_loss += loss.item()
        loss_count += 1
        


        if _%100==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
        
        if _%1000<0 or _==len(loader):
          # Save model checkpoint
          output_dir = os.path.join("sliding_t5_450_bs=4","epoch-{}".format(epoch),'checkpoint_step-{}'.format(_))
          if not os.path.exists(output_dir):
              os.makedirs(output_dir)
          model_to_save = model.module if hasattr(model,'module') else model  # Take care of distributed/parallel training
          model_to_save.save_pretrained(output_dir)
          tokenizer.save_pretrained(output_dir)
          #torch.save(args, os.path.join(output_dir, 'training_args.bin'))
          print("model saved at step {0}".format(_))

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()


        # xm.optimizer_step(optimizer)
        # xm.mark_step()
    output_dir = os.path.join("new", "baseline_my450","epoch-{}".format(epoch + 20),'end-stage')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model,'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    #torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    print("model saved at end epoch")
    print("avarage_loss = {0}".format(avg_loss / loss_count))
    np.savetxt('new/baseline_my450//losses_{0}.txt'.format(str(time.time())),losses)
    #print(losses)
	





def main():

    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("new/baseline_my450/epoch-19/end-stage/", use_cdn = False, cache_dir="new_cache_dir/")
    #tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", use_cdn = False, cache_dir="new_cache_dir/")

    



    #df = pd.read_csv('news_summary.csv',encoding='latin-1')
    df = pd.read_csv('my_format_450_train.csv',encoding='latin-1')
    df = df[['text','ctext']]

    df.ctext = df.ctext
    print(df.head())


    train_size = 1
    train_dataset=df.sample(frac=train_size)
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, 450, 450)
    val_set = CustomDataset(val_dataset, tokenizer, 450, 450)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
        }

    val_params = {
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    

    #model = T5ForConditionalGeneration.from_pretrained("model/epoch-4/checkpoint_step-2000/", use_cdn = False, cache_dir="new_cache_dir/")
    #config = AutoConfig.from_pretrained('t5-base', dropout_rate = 0.3)
    #model = T5ForConditionalGeneration.from_pretrained("model_760_with_text/epoch-8/checkpoint_step-2000/",use_cdn = False, cache_dir="new_cache_dir/")
    #model.config.__dict__['dropout_rate'] = 0.3
    config = T5Config(decoder_start_token_id=2)
    #model = T5ForConditionalGeneration(config = config)#, use_cdn = False, cache_dir="new_cache_dir/")
    model = T5ForConditionalGeneration.from_pretrained("new/baseline_my450/epoch-19/end-stage/",config=config,use_cdn = False, cache_dir="new_cache_dir/")
    #model = TFT5ForConditionalGeneration(config)
    #model = BartForConditionalGeneration.from_pretrained("facebook/bart-base" ,use_cdn = False, cache_dir="new_cache_dir/")
    
    #model.config.__dict__['dropout'] = 0.3
    print(model.config)
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    #optimizer = torch.optim.Adam(params =  model.parameters(), lr=1e-4)
    optimizer = AdamW(params = model.parameters(), lr=1e-5, weight_decay = 0.01, eps = 1e-6)


    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in range(50):
        train(epoch, tokenizer, model, device, training_loader, optimizer)


if __name__ == '__main__':
    main()
	
	

