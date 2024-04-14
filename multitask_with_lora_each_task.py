'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
from itertools import zip_longest

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask
from peft import LoraModel, LoraConfig
import os

TQDM_DISABLE=False
writer = SummaryWriter('runs/multitask_new_training')

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config, lora_config_sentiment=None, lora_config_paraphrase=None, lora_config_similarity=None):  
        # Add lora_config as an optional argument
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # If LoRA config is provided, initialize separate LoRA layers for each task with their specific configurations
        if lora_config_sentiment is not None:
            self.lora_sentiment = LoraModel(self.bert, lora_config_sentiment, "default") 
        if lora_config_paraphrase is not None:
            self.lora_paraphrase = LoraModel(self.bert, lora_config_paraphrase, "default") 
        if lora_config_similarity is not None:
            self.lora_similarity = LoraModel(self.bert, lora_config_similarity, "default") 

        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        # Task-specific layers:
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.paraphrase_classifier = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)
        self.similarity_classifier = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)

        # You can also add dropout layers if necessary
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids, attention_mask, task):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        
        # Route the input through the appropriate LoRA-enhanced BERT model based on the task
        if task == 'sentiment':
            outputs = self.lora_sentiment(input_ids=input_ids, attention_mask=attention_mask)
        elif task == 'paraphrase':
            outputs = self.lora_paraphrase(input_ids=input_ids, attention_mask=attention_mask)
        elif task == 'similarity':
            outputs = self.lora_similarity(input_ids=input_ids, attention_mask=attention_mask)
        else:
            raise ValueError("Invalid task specified.")

        # Extract the [CLS] token's embeddings
        cls_embeddings = outputs['last_hidden_state'][:, 0, :]
        
        # Apply dropout to the [CLS] embeddings
        cls_embeddings = self.dropout(cls_embeddings)
        
        # Return the [CLS] embeddings
        return cls_embeddings


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        # Get the embeddings for the input sentences
        cls_embeddings = self.forward(input_ids, attention_mask, task = 'sentiment')
        
        # Apply the sentiment classifier to get logits for each sentiment class
        sentiment_logits = self.sentiment_classifier(cls_embeddings)
        
        return sentiment_logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        # Get the embeddings for the first set of sentences
        cls_embeddings_1 = self.forward(input_ids_1, attention_mask_1, task = 'paraphrase')
        # Get the embeddings for the second set of sentences
        cls_embeddings_2 = self.forward(input_ids_2, attention_mask_2, task = 'paraphrase')
        
        # Concatenate the embeddings from both sentences
        combined_embeddings = torch.cat((cls_embeddings_1, cls_embeddings_2), dim=1)
        
        # Ensure the paraphrase classifier linear layer matches the combined embedding size
        if self.paraphrase_classifier.in_features != combined_embeddings.size(1):
            # Adjust the classifier's input features
            self.paraphrase_classifier = nn.Linear(combined_embeddings.size(1), 1).to(combined_embeddings.device)
        
        # Apply the paraphrase classifier to get a single logit for each sentence pair
        paraphrase_logit = self.paraphrase_classifier(combined_embeddings)
        
        return paraphrase_logit


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        # Get the embeddings for the first set of sentences
        cls_embeddings_1 = self.forward(input_ids_1, attention_mask_1, task = 'similarity')
        # Get the embeddings for the second set of sentences
        cls_embeddings_2 = self.forward(input_ids_2, attention_mask_2, task = 'similarity')

        combined_embeddings = torch.cat((cls_embeddings_1, cls_embeddings_2), dim=1)
        
        
        # Apply the similarity classifier to get a single logit for each sentence pair
        similarity_logit = self.similarity_classifier(combined_embeddings)
        
        return similarity_logit


def save_model(model, optimizers, args, config, filepath):

     # Convert config to a dictionary if it's not already one
    if not isinstance(config, dict):
        config_dict = vars(config)  # vars() works well with SimpleNamespace and similar objects

    # Prepare the optimizers' state dictionaries
    optimizers_state = {}
    for name, optimizer in optimizers.items():
        optimizers_state[name] = optimizer.state_dict()

    save_info = {
        'model': model.state_dict(),
        # 'optim': optimizer.state_dict(),
        'optimizers': optimizers_state,  # Save all optimizers' states
        'args': args,
        'model_config': config_dict,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT on three tasks simultaneously.'''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    print(device)

    # Load the data for all three tasks.
    sst_train_data, _, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

    # Prepare dataloaders for all datasets
    sst_train_dataset = SentenceClassificationDataset(sst_train_data, args)
    sst_train_dataloader = DataLoader(sst_train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_dataset.collate_fn)

    sst_dev_dataset = SentenceClassificationDataset(sst_dev_data, args)
    sst_dev_dataloader = DataLoader(sst_dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_dataset.collate_fn)

    para_train_dataset = SentencePairDataset(para_train_data, args)
    para_train_dataloader = DataLoader(para_train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_dataset.collate_fn)

    para_dev_dataset = SentencePairDataset(para_dev_data, args)
    para_dev_dataloader = DataLoader(para_dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_dataset.collate_fn)

    sts_train_dataset = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_dataset.collate_fn)

    sts_dev_dataset = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_dev_dataloader = DataLoader(sts_dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_dataset.collate_fn)

    # Init model, optimizer, and other training components.
    config = {
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'hidden_size': BERT_HIDDEN_SIZE,
        'data_dir': '.',
        'option': args.option
    }
    config = SimpleNamespace(**config)

    # Task-specific LoRA configurations
    lora_config_sentiment = LoraConfig(r=4, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.1)
    lora_config_paraphrase = LoraConfig(r=2, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.05)
    lora_config_similarity = LoraConfig(r=8, lora_alpha=16, target_modules=["value"], lora_dropout=0.1)

    # Initialize your model with separate LoRA configurations for each task
    model = MultitaskBERT(config, 
                          lora_config_sentiment=lora_config_sentiment, 
                          lora_config_paraphrase=lora_config_paraphrase, 
                          lora_config_similarity=lora_config_similarity)

    model = model.to(device)
    
    # define task specific parameters that will be updated 
    sentiment_params = list(model.sentiment_classifier.parameters()) + list(model.lora_sentiment.parameters())
    paraphrase_params = list(model.paraphrase_classifier.parameters()) + list(model.lora_paraphrase.parameters())
    similarity_params = list(model.similarity_classifier.parameters()) + list(model.lora_similarity.parameters())

    # Create optimizers for shared and task-specific parameters including LoRA parameters
    sentiment_optimizer = AdamW(sentiment_params, lr=args.lr)
    paraphrase_optimizer = AdamW(paraphrase_params, lr=args.lr)
    similarity_optimizer = AdamW(similarity_params, lr=args.lr)

    best_sentiment_accuracy = 0.0
    best_paraphrase_accuracy = 0.0
    best_sts_corr = -1.0  # Pearson correlation ranges from -1 to 1

    # Initialize losses for each task
    train_loss = 0
    num_batches = 0
    max_dataset_size = max(len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader))

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()

        # Wrap the zipped dataloaders with tqdm for a progress bar
        progress_bar = tqdm(zip_longest(sst_train_dataloader, para_train_dataloader, sts_train_dataloader), 
                            total=max_dataset_size,
                            desc=f"Epoch {epoch+1}/{args.epochs}")

        # Interleave batches from each dataset
        for i, batch in enumerate(progress_bar, 1):
            sentiment_optimizer.zero_grad()
            paraphrase_optimizer.zero_grad()
            similarity_optimizer.zero_grad()
            
            # Initialize the batch loss
            batch_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            # Unpack batches
            batch_sst, batch_para, batch_sts = batch

            # Handle sentiment analysis batch
            if batch_sst is not None:
                b_ids, b_mask, b_labels = batch_sst['token_ids'], batch_sst['attention_mask'], batch_sst['labels']
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device).float()
                b_labels = b_labels.to(device)

                sentiment_logits = model.predict_sentiment(b_ids, b_mask).float()
                current_batch_size_float = torch.tensor(b_labels.size(0), dtype=torch.float32, device=device)
                loss_sst = F.cross_entropy(sentiment_logits, b_labels.view(-1), reduction='sum') / current_batch_size_float
                batch_loss += loss_sst

                loss_sst.backward()
                sentiment_optimizer.step()  # Update sentiment-specific parameters
                
            # Handle paraphrase detection batch
            if batch_para is not None:
                (b_ids1, b_mask1,
                 b_ids2, b_mask2,
                 b_labels) = (batch_para['token_ids_1'], batch_para['attention_mask_1'],
                              batch_para['token_ids_2'], batch_para['attention_mask_2'],
                              batch_para['labels'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device).float()
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device).float()
                b_labels = b_labels.to(device)

                paraphrase_logit = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2).float()
                current_batch_size_float = torch.tensor(b_labels.size(0), dtype=torch.float32, device=device)
                loss_para = F.binary_cross_entropy_with_logits(paraphrase_logit.squeeze(), b_labels.float(), reduction='sum') / current_batch_size_float
                batch_loss += loss_para

                loss_para.backward()
                paraphrase_optimizer.step()  # Update paraphrase-specific parameters

            # Handle STS batch
            if batch_sts is not None:
                (b_ids1, b_mask1,
                 b_ids2, b_mask2,
                 b_labels) = (batch_sts['token_ids_1'], batch_sts['attention_mask_1'],
                              batch_sts['token_ids_2'], batch_sts['attention_mask_2'],
                              batch_sts['labels'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device).float()
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device).float()
                b_labels = b_labels.to(device).float()

                similarity_logit = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2).float()
                current_batch_size_float = torch.tensor(b_labels.size(0), dtype=torch.float32, device=device)
                loss_sts = F.mse_loss(similarity_logit.squeeze(), b_labels.float(), reduction='sum') / current_batch_size_float
                batch_loss += loss_sts

                loss_sts.backward()
                similarity_optimizer.step()  # Update similarity-specific parameters

            train_loss += batch_loss.item()
            num_batches += 1
            avg_loss = train_loss / i

            # Update progress bar description with the current average loss
            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs} Avg Loss: {avg_loss:.4f}")

            # Calculate the average loss at this point in training
            writer.add_scalar('Training Loss', avg_loss, epoch * max_dataset_size + i)

        # At the end of each epoch, log the average loss for the epoch
        epoch_avg_loss = train_loss / num_batches
        writer.add_scalar('Average Training Loss per Epoch', epoch_avg_loss, epoch)

        eval_results = model_eval_multitask(sst_dev_dataloader,
                                            para_dev_dataloader,
                                            sts_dev_dataloader,
                                            model,
                                            device)
        
        (sentiment_accuracy, sst_y_pred, sst_sent_ids,
         paraphrase_accuracy, para_y_pred, para_sent_ids,
         sts_corr, sts_y_pred, sts_sent_ids) = eval_results

        # Check if there is an improvement in sentiment analysis accuracy
        if sentiment_accuracy > best_sentiment_accuracy:
            best_sentiment_accuracy = sentiment_accuracy
            # save_model(model, optimizer, args, config, f"{args.filepath}")
            save_model(model, {'sentiment': sentiment_optimizer, 
                               'paraphrase': paraphrase_optimizer, 
                               'similarity': similarity_optimizer}, args, config, f"{args.filepath}")

        # Check if there is an improvement in paraphrase detection accuracy
        if paraphrase_accuracy > best_paraphrase_accuracy:
            best_paraphrase_accuracy = paraphrase_accuracy
            save_model(model, {'sentiment': sentiment_optimizer, 
                               'paraphrase': paraphrase_optimizer, 
                               'similarity': similarity_optimizer}, args, config, f"{args.filepath}")

        # Check if there is an improvement in STS Pearson correlation
        if sts_corr > best_sts_corr:
            best_sts_corr = sts_corr
            save_model(model, {'sentiment': sentiment_optimizer, 
                               'paraphrase': paraphrase_optimizer, 
                               'similarity': similarity_optimizer}, args, config, f"{args.filepath}")

        # Design print statements to output all result variables
        print(f"dev set evaluation results:")
        print(f"Sentiment Analysis Accuracy: {sentiment_accuracy:.4f}")
        print(f"Sample Sentiment Predictions: {sst_y_pred[:5]}")  # Print the first 5 predictions as a sample
        print(f"Sample Sentiment IDs: {sst_sent_ids[:5]}")        # Print the first 5 IDs as a sample

        print(f"Paraphrase Detection Accuracy: {paraphrase_accuracy:.4f}")
        print(f"Sample Paraphrase Predictions: {para_y_pred[:5]}")  # Print the first 5 predictions as a sample
        print(f"Sample Paraphrase IDs: {para_sent_ids[:5]}")        # Print the first 5 IDs as a sample

        print(f"STS (Semantic Textual Similarity) Pearson Correlation: {sts_corr:.4f}")
        print(f"Sample STS Predictions: {sts_y_pred[:5]}")           # Print the first 5 predictions as a sample
        print(f"Sample STS IDs: {sts_sent_ids[:5]}")                 # Print the first 5 IDs as a sample
    
    writer.close()  # Close the TensorBoard writer


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

       # Load the saved model checkpoint
        saved = torch.load(args.filepath)

        # Retrieve the saved model configuration
        saved_config = saved.get('model_config', {})

        # Ensure the config object has the num_labels attribute from the saved configuration
        config = SimpleNamespace(**saved_config)
        print(f'config: {config}')

        # Task-specific LoRA configurations
        lora_config_sentiment = LoraConfig(r=4, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.1)
        lora_config_paraphrase = LoraConfig(r=2, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.05)
        lora_config_similarity = LoraConfig(r=8, lora_alpha=16, target_modules=["value"], lora_dropout=0.1)

        # Initialize your model with separate LoRA configurations for each task
        model = MultitaskBERT(config, 
                            lora_config_sentiment=lora_config_sentiment, 
                            lora_config_paraphrase=lora_config_paraphrase, 
                            lora_config_similarity=lora_config_similarity)

        model.load_state_dict(saved['model'])
        model = model.to(device)

        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv") 

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output-lora.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output-lora.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output-lora.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output-lora.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output-lora.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output-lora.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask-lora.pt' # Save path
    seed_everything(args.seed)  # Fix the seed for reproducibility
    train_multitask(args)
    test_multitask(args)