import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        #super(DecoderRNN, self).__init__()
        super().__init__()
       
        # set class variables
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        
        # define model layers
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
  
        # Linear layer maps hidden_size to scores of vocab_size
        self.hidden2scores = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        # get batch size
        batch_size = features.size(0)

        # get embeddings for captions except last one
        capt_embeds = self.embed(captions[:,:-1])
                
        # concatenate features and embedded captions
        inputs = torch.cat((features.unsqueeze(1), capt_embeds),1)
         
        # clean out hidden state and cell state
        if (torch.cuda.is_available()):
            hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda(),
                      torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
        else:
            hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                      torch.zeros(self.num_layers, batch_size, self.hidden_size))
            
        lstm_out, hidden = self.lstm(inputs, hidden)
        
        # score outputs
        out = self.hidden2scores(lstm_out)
              
        # return output word scores
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        predicted_word_ids = []
        input = inputs
                
        # clean out hidden state and cell state
        if (torch.cuda.is_available()):
            hidden = (torch.zeros(self.num_layers, 1, self.hidden_size).cuda(),
                      torch.zeros(self.num_layers, 1, self.hidden_size).cuda())
        else:
            hidden = (torch.zeros(self.num_layers, 1, self.hidden_size),
                      torch.zeros(self.num_layers, 1, self.hidden_size))
                    
        for _ in range(max_len):
            
            lstm_out, hidden = self.lstm(input, hidden)
            
            # score outputs
            out = self.hidden2scores(lstm_out)
                        
            # get word id with max probability
            _, word_id = out.max(dim=2)
            word_id_int = word_id.item()
                       
            # append word id to list of predictions
            predicted_word_ids.append(word_id_int)
            
            # if predicted word is 1 (<end>) then stop
            if word_id_int == 1:
                break
            
            # embedding of last word id becomes next input 
            input = self.embed(word_id) 
                
        return predicted_word_ids