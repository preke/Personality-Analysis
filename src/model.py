from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel, RobertaTokenizer, RobertaForSequenceClassification
from torch.nn import TransformerEncoder
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import numpy as np




class RobertaClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks.
    """

    def __init__(self, config, num_labels):
        super().__init__()        
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        self.out_proj = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Baseline_1_roberta(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels      = 2
        self.config          = config
        self.roberta         = RobertaModel(config)
        self.personality_cls = RobertaClassificationHead(config, num_labels=self.num_labels)
        self.get_vad         = RobertaClassificationHead(config, num_labels=3)
        self.init_weights()
    
    def forward(self, text, token_type_ids=None, attention_mask=None):  
        if token_type_ids:
            outputs   = self.roberta(text, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            outputs   = self.roberta(text, attention_mask=attention_mask)
        embedding = outputs[1]
        logit_p   = self.personality_cls(embedding)
        logit_vad = self.get_vad(embedding)
        return logit_p, logit_vad
    



class Baseline_1(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels      = 2
        self.config          = config
        self.bert            = BertModel(config)
        self.personality_cls = nn.Linear(config.hidden_size, self.num_labels)
        self.get_vad         = nn.Linear(config.hidden_size, 3)  # 3 for vad
        self.init_weights()
    
    def forward(self, text, token_type_ids=None, attention_mask=None):  
        if token_type_ids:
            outputs   = self.bert(text, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            outputs   = self.bert(text, attention_mask=attention_mask)
        embedding = outputs[1]
        logit_p   = self.personality_cls(embedding)
        logit_vad = self.get_vad(embedding)
        return logit_p, logit_vad
    

class Baseline_2(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels      = 2
        self.config          = config
        self.bert            = BertModel(config)
        self.personality_cls = nn.Linear(config.hidden_size, self.num_labels)
        self.get_vad         = nn.Linear(config.hidden_size, 3)  # 3 for vad
        self.init_weights()
    

    def forward(self, uttr, uttr_mask, dialog, dialog_mask, dialog_seg_embeddings):  
        
        # print(self.config)
        
        uttr_outputs  = self.bert(uttr, attention_mask=uttr_mask)
        uttr_embedding = uttr_outputs[1]
        logit_vad = self.get_vad(uttr_embedding)

        dialog_outputs  = self.bert(dialog, token_type_ids=dialog_seg_embeddings, attention_mask=dialog_mask)
        dialog_embedding = dialog_outputs[1]
        logit_p   = self.personality_cls(dialog_embedding)
        return logit_p, logit_vad   
    

    
class Uttr_VAD_embedding(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels      = 2
        self.config          = config
        self.bert            = BertModel(config)
        self.vad_to_embed    = nn.Linear(3, config.hidden_size)
        self.personality_cls = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, uttr, attention_mask, segment_embeddings=None):  
        
        uttr_input_embeddings = self.vad_to_embed(uttr)
        if segment_embeddings != None:
            uttr_outputs  = self.bert(attention_mask=attention_mask, inputs_embeds=uttr_input_embeddings, token_type_ids=segment_embeddings)
        else:
            uttr_outputs  = self.bert(attention_mask=attention_mask, inputs_embeds=uttr_input_embeddings)
        uttr_embedding = uttr_outputs[1]
        logit_p = self.personality_cls(uttr_embedding)
        return logit_p

class Uttr_VAD_embedding_lookup(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels      = 2
        self.config          = config
        self.bert            = BertModel(config)
        self.vad_to_embed    = nn.Linear(3, config.hidden_size)
        self.personality_cls = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, uttr, attention_mask, segment_embeddings=None):  
        
        uttr_input_embeddings = self.vad_to_embed(uttr)
        if segment_embeddings != None:
            uttr_outputs  = self.bert(attention_mask=attention_mask, inputs_embeds=uttr_input_embeddings, token_type_ids=segment_embeddings)
        else:
            uttr_outputs  = self.bert(attention_mask=attention_mask, inputs_embeds=uttr_input_embeddings)
        uttr_embedding = uttr_outputs[1]
        logit_p = self.personality_cls(uttr_embedding)
        return logit_p
    
    

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x, dialog_states):
        out = self.attention(x, dialog_states)
        out = self.feed_forward(out)
        return out


# class Dialog_State_Encoding(nn.Module):
#     def __init__(self, embed, pad_size, dropout, device):
#         super(Dialog_State_Encoding, self).__init__()
#         self.device = device
        
#         # calculate the dialog state encoding

#         self.dim_model = embed # 32
#         self.pad_size = pad_size # 30
#         self.dropout = nn.Dropout(dropout)
#         self.seg_embedding  = nn.Linear(1, self.dim_model)

    
#     def forward(self, x, dialog_states):
#         dialog_states = dialog_states.view(-1,1).float() # (batch_size*pad_size)*1
#         state_emd = self.seg_embedding(dialog_states) # (batch_size*pad_size)* 32
#         state_emd = state_emd.view(-1,self.pad_size,self.dim_model) # batch_size * 30 * 32
        
#         out = x + nn.Parameter(state_emd, requires_grad=False).to(self.device)
#         out = self.dropout(out)
#         return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Context_Encoder(nn.Module):
    def __init__(self, args):
        super(Context_Encoder, self).__init__()

        self.args        = args
        self.pad_size    = args.MAX_NUM_UTTR
        self.dropout     = args.drop_out
        self.num_head    = 1
        self.dim_model   = args.d_transformer # 32
        self.num_encoder = 1
        self.num_classes = args.num_class
        self.device      = args.device
        self.hidden      = 512

        self.position_embedding = Positional_Encoding(embed=self.dim_model, pad_size=self.pad_size, dropout=self.dropout, device=self.device)
        # self.dialog_state_embedding = Dialog_State_Encoding(embed=self.dim_model, pad_size=self.pad_size, dropout=self.dropout, device=self.device)
        self.encoder = Encoder(dim_model=self.dim_model, num_head=self.num_head, hidden=self.hidden, dropout=self.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(self.num_encoder)]) # num_encoder

        # self.fc1 = nn.Linear(self.pad_size * self.dim_model, self.num_classes)
        self.fc1 = nn.Linear(self.dim_model, self.num_classes)
        
    def forward(self, x, dialog_states, d_transformer, args):
        out = x.view(-1, self.pad_size, self.dim_model) # batch_size * context_len * 32
        print(out.shape)

        out = self.position_embedding(out)

        print(out.shape)
        ## add dialog state
        # out = self.dialog_state_embedding(out, dialog_states)

        for encoder in self.encoders:
            out = encoder(out, dialog_states)
        # out = out.view(out.size(0), -1)
        out = torch.mean(out, 1)
        out = self.fc1(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale, mask):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        
        mask = mask.unsqueeze(-1).expand(-1, -1, attention.shape[1])
        attention = attention * scale
        attention = attention.masked_fill_(mask == -1, -1e9)


        # print(attention)
        # import time
        # time.sleep(100)

        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x, dialog_states):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        print(Q.shape, K.shape, V.shape)
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale, dialog_states)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.5):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class DialogVAD(BertPreTrainedModel):
    
    def __init__(self, config, args):
        super().__init__(config)
        self.args            = args
        self.num_labels      = args.num_class
        self.d_transformer   = args.d_transformer
        self.config          = config
        self.bert            = BertModel(config)
        # self.get_vad         = nn.Linear(config.hidden_size, 3)  # 3 for vad
        self.reduce_size     = nn.Linear(config.hidden_size, self.d_transformer) # from 768 reduce to 64 for the appended Transformer model


        self.context_encoder = Context_Encoder(args)
        # self.personality_cls     = nn.Linear(config.hidden_size, 2) # binary classification
        self.init_weights()
    
    def forward(self, context, context_mask, dialog_states):  
        
        

        batch_size, max_ctx_len, max_utt_len = context.size() # 16 * 30 * 32

        context_utts = context.view(max_ctx_len, batch_size, max_utt_len)    # [batch_size * dialog_length * max_uttr_length]122
        context_mask = context_mask.view(max_ctx_len, batch_size, max_utt_len)   

        uttr_outputs  = [self.bert(uttr, uttr_mask) for uttr, uttr_mask in zip(context_utts,context_mask)]
        # 30 * 16 * 768 

        uttr_outputs = [uttr_output[1] for uttr_output in uttr_outputs] # 30 * 16 * 768 
        uttr_embeddings = torch.stack([self.reduce_size(uttr_output) for uttr_output in uttr_outputs]) # 30 * 16 * 32
        uttr_embeddings = torch.autograd.Variable(uttr_embeddings.view(batch_size, max_ctx_len, self.d_transformer), requires_grad=True)
        # ## vad regression
        # # logit_vads      = self.get_vad(uttr_embedding).view(-1, 10, 3) # [batch_size * dialog_length * 3]
        
        # ---- concat with transformer to do the self-attention
        logits = self.context_encoder(uttr_embeddings, dialog_states, self.d_transformer, self.args) # [batch_size * 2]

        return logits#, logit_vads



class DialogVAD_roberta(RobertaPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels      = 2
        self.config          = config
        self.get_vad         = RobertaClassificationHead(config, num_labels=3)
        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.context_encoder = Context_Encoder(config)
        self.personality_cls = RobertaClassificationHead(config, num_labels=self.num_labels)
        self.init_weights()
    
    def forward(self, context, utts_attn_mask, dialog_states):  

        '''
        context: [batchsize * dialog_length * max_uttr_length]
        dialog_states: [batchsize * dialog_length] indicate if the current utterance is from the analyzed speaker 
        '''

        batch_size, max_ctx_len, max_utt_len = context.size()
        # print(batch_size, max_ctx_len, max_utt_len)


        utts = context.view(-1, max_utt_len)    # [batch_size * dialog_length * max_uttr_length]
        utts_attn_mask = utts_attn_mask.view(-1, max_utt_len)    
        
        # print(utts.shape)
        # print(utts_attn_mask.shape)
        
        uttr_outputs  = self.roberta(input_ids=utts, attention_mask=utts_attn_mask)
        uttr_embedding = uttr_outputs[1]
        # print(uttr_embedding.shape)
        
        ## vad regression
        logit_vads      = self.get_vad(uttr_embedding).view(-1, 10, 3) # [batch_size * dialog_length * 3]
        # print(logit_vads.shape)
        

        # ---- concat with transformer to do the self-attention
        logits = self.context_encoder(uttr_embedding, dialog_states) # [batch_size * 2]
        # print(logit.size())

        return logits, logit_vads




        
        
        
        
        
        
        
        
        
        