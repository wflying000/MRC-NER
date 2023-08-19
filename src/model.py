import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class FeedForward(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size, act_func="gelu", dropout=0.1):
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.intermediate_size = intermediate_size
        self.output_size = output_size
        self.classifier1 = nn.Linear(input_size, intermediate_size)
        self.classifier2 = nn.Linear(intermediate_size, output_size)
        self.dropout = nn.Dropout(dropout)
        if act_func == "gelu":
            self.act_func = F.gelu
        elif act_func == "relu":
            self.act_func = F.relu
        elif act_func == "tanh":
            self.act_func = F.tanh
        else:
            raise ValueError

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = self.act_func(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2


class MRCNERModel(nn.Module):
    def __init__(self, config):
        super(MRCNERModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(config.pretrained_model_path)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_outputs = FeedForward(
            input_size=config.hidden_size * 2, 
            intermediate_size=config.intermediate_size,
            output_size=1, 
            dropout=config.dropout,
        )
        self.hidden_size = config.hidden_size

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        encoder_output = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        hidden_state = encoder_output.last_hidden_state  # (batch, seq_len, hidden_size)
        batch_size, seq_len, hid_size = hidden_state.size()

        start_logits = self.start_outputs(hidden_state).squeeze(-1)  # (batch, seq_len)
        end_logits = self.end_outputs(hidden_state).squeeze(-1)  # (batch, seq_len)
        start_extend = hidden_state.unsqueeze(2).expand(-1, -1, seq_len, -1) # (batch, seq_len, seq_len, hidden_size)
        end_extend = hidden_state.unsqueeze(1).expand(-1, seq_len, -1, -1) # (batch, seq_len, seq_len, hidden_size)
        span_matrix = torch.cat([start_extend, end_extend], 3) # (batch, seq_len, seq_len, hidden_size * 2)
        span_logits = self.span_outputs(span_matrix).squeeze(-1) # (batch, seq_len, seq_len)

        outputs = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "span_logits": span_logits,
        }

        return outputs