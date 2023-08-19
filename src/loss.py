import torch

class MRCNERLoss():
    def __init__(
        self, 
        start_weight=1,
        end_weight=1,
        span_weight=1,
        span_loss_candidate="all", 
        eps=1e-10
    ):
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.span_weight = span_weight
        self.span_loss_candidate = span_loss_candidate
        self.eps = eps
        self.loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")

    def __call__(
        self,
        start_logits,
        end_logits,
        span_logits,
        start_labels,
        end_labels,
        span_labels,
        start_label_mask,
        end_label_mask,
    ):
        batch_size, seq_len = start_logits.size()

        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        span_label_row_mask = start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
        span_label_col_mask = end_label_mask.unsqueeze(-2).expand(-1, seq_len, -1)
        span_label_mask = span_label_row_mask & span_label_col_mask
        span_label_mask = torch.triu(span_label_mask, 0)

        if self.span_loss_candidate == "all":
            span_float_label_mask = span_label_mask.view(batch_size, -1).float()
        else:
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            
            if self.span_loss_candidate == "gold":
                span_candidate = ((start_labels.unsqueeze(-2).expand(-1, -1, seq_len) > 0) &
                                  (end_labels.unsqueeze(-1).expand(-1, seq_len, -1) > 0))
            elif self.span_loss_candidate == "pred_gold_random":
                pred_gold = torch.logical_or(
                    (start_preds.unsqueeze(-2).expand(-1, -1, seq_len) &
                     end_preds.unsqueeze(-1).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-2).expand(-1, -1, seq_len) &
                     end_labels.unsqueeze(-1).expand(-1, seq_len, -1))
                )
                data_generator = torch.Generator()
                data_generator.manual_seed(0)
                random_tensor = torch.empty(batch_size, seq_len, seq_len).uniform_(0, 1)
                random_tensor = torch.bernoulli(random_tensor, generator=data_generator).long()
                random_tensor = random_tensor.to(pred_gold.device)
                span_candidate = torch.logical_or(
                    pred_gold, random_tensor
                )
            else:
                span_candidate = torch.logical_or(
                    (start_preds.unsqueeze(-2).expand(-1, -1, seq_len) &
                     end_preds.unsqueeze(-1).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-2).expand(-1, -1, seq_len) &
                     end_labels.unsqueeze(-1).expand(-1, seq_len, -1))
                )
            span_label_mask = span_label_mask & span_candidate
            span_float_label_mask = span_label_mask.view(batch_size, -1).float()
            
        start_loss = self.loss_fct(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = self.loss_fct(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        span_loss = self.loss_fct(span_logits.view(batch_size, -1), span_labels.view(batch_size, -1).float())
        span_loss = (span_loss * span_float_label_mask).sum() / (span_float_label_mask.sum() + self.eps)

        loss = self.start_weight * start_loss + \
               self.end_weight * end_loss + \
               self.span_weight * span_loss

        return loss, start_loss, end_loss, span_loss
    






        