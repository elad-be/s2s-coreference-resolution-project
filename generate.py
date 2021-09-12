print("start t5 ")
# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BartForConditionalGeneration, BartTokenizer

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, get_linear_schedule_with_warmup
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
        #ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        #text = ' '.join(text.split())

        word_list = text.split()
        text_len = len(word_list)
        
        word_list = ctext.split()
        ctext_len = len(word_list)
        print("number of words {0}",format(ctext_len))
        
        
        if ctext_len >1050:
            text_len = 1050
            ctext_len = 1050
            
        print("----------original text----------")
        print(ctext)
        source = self.tokenizer.batch_encode_plus([ctext],  max_length= 1000 ,pad_to_max_length=True, truncation=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text],  max_length= 1000, pad_to_max_length=True, truncation=True,return_tensors='pt')
        #print(tokenizer.decode(source.input_ids[0], clean_up_tokenization_spaces=False))
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
	

def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=len(ids[0]), 
                num_beams=4,
                #repetition_penalty=2.5, 
                #length_penalty=0.6, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)for t in y]
            print("----------generated text----------")
            print(preds)
            print("----------label text----------")
            print(target)
            print("######")
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals
	





def main(file_name, ep, save_name):
    proxies = {"http": "http://10.10.1.10:3128","https": "https://10.10.1.10:1080",}

    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    
    ###T5
    tokenizer = T5Tokenizer.from_pretrained(file_name, use_cdn = False, cache_dir="new_cache_dir/")
    
    
    #### BART
    #tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", use_cdn = False, cache_dir="new_cache_dir/")
    #tokenizer = BartTokenizer.from_pretrained(file_name, use_cdn = False, cache_dir="new_cache_dir/")

    
    
    
    #df = pd.read_csv('testNoLimitwithText.csv',encoding='latin-1')
    #df = pd.read_csv('real_devnolimits.csv',encoding='latin-1')
    #df = pd.read_csv('my_format_dev.csv',encoding='latin-1')
    df = pd.read_csv('testNoLimitFix121.csv',encoding='latin-1')
    #df = pd.read_csv('dev450.csv',encoding='latin-1')
    
    #df = pd.read_csv('devPerDocNoLimit.csv',encoding='latin-1')
    #df = pd.read_csv('dev450.csv',encoding='latin-1')
    df = df[['text','ctext']]
    #df.ctext = 'summarize: ' + df.ctext
    df.ctext = df.ctext
    print(df.head())

    

    train_size = 1
    val_dataset=df.sample(frac=train_size)
    #val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    #train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    #print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))


    # Creating the Training and Validation dataset for further creation of Dataloader
    #training_set = CustomDataset(train_dataset, tokenizer, 450, 450)
    val_set = CustomDataset(val_dataset, tokenizer, 450, 450)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
        }

    val_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    #training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)



    ####    T5
    config = T5Config(decoder_start_token_id=2)
    model = T5ForConditionalGeneration.from_pretrained(file_name, config=config,use_cdn = False, cache_dir="new_cache_dir/")
    
    ###   BART
    #model = BartForConditionalGeneration.from_pretrained("bart_450_run4/epoch-10/checkpoint_step-5000/" ,use_cdn = False, cache_dir="new_cache_dir/")
    #model = BartForConditionalGeneration.from_pretrained("facebook/bart-base" ,use_cdn = False, cache_dir="new_cache_dir/")
    #model = BartForConditionalGeneration.from_pretrained(file_name ,use_cdn = False, cache_dir="new_cache_dir/")
    
    #model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    #optimizer = torch.optim.Adam(params =  model.parameters(), lr=1e-4)
    optimizer = AdamW(params = model.parameters(), lr=1e-5, weight_decay = 0.01)

    # Log metrics with wandb
    #wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')



    for epoch in range(1):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        output_dir = os.path.join('test_eval/{0}.csv'.format(save_name)) # os.path.join("sliding_t5_450_pred_no_spaces",'sliding_t5_450_epoch_{0}_no_limit_beam=4.csv'.format(ep))
        final_df.to_csv(output_dir)
        print('Output Files generated for review')

if __name__ == '__main__':
    file_name = ["new/baseline_my450/epoch-50/end-stage/"]
    save_name = ['pred_myformat_baseline50_450_beam4']
    for i in range(len(file_name)):
        print("epoch number {0}".format(i))
        main(file_name[i], i, save_name[i])
	
	
