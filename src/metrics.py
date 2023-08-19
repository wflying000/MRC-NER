import torch

class MRCNERMetrics():
    def __init__(self, entity_types):
        self.entity_types = entity_types
        self._init_category()

    def clear(self):
        self._init_category()

    def _init_category(self):
        self.category = self._create_category()

    def _create_category(self):
        category = {}
        for ent in self.entity_types:
            category[ent] = {}
            category[ent]["TP"] = 0
            category[ent]["FP"] = 0
            category[ent]["FN"] = 0
        return category
    
    def add_batch(self, start_logits, end_logits, span_logits, start_label_mask, end_label_mask, span_labels, entity_types):
        category = self._compute_category_count(start_logits, end_logits, span_logits, start_label_mask, end_label_mask, span_labels, entity_types)
        self._update_category(category)

    def compute(self):
        category = self._compute_category_metrics(self.category)
        overall = self._compute_overall_metrics(category)
        metrics = {
            "category": category,
            "overall": overall,
        }
        return metrics

    def _update_category(self, category):
        for ent in category:
            self.category[ent]["TP"] += category[ent]["TP"]
            self.category[ent]["FP"] += category[ent]["FP"]
            self.category[ent]["FN"] += category[ent]["FN"]
    
    def _compute_category_count(self, start_logits, end_logits, span_logits, start_label_mask, end_label_mask, span_labels, entity_types):
        category = self._create_category()
        start_label_mask = start_label_mask.bool()
        end_label_mask = end_label_mask.bool()
        span_labels = span_labels.bool()
        bsz, seq_len = start_label_mask.size()
        span_preds = span_logits > 0
        start_preds = start_logits > 0
        end_preds = end_logits > 0
        start_preds = start_preds.bool()
        end_preds = end_preds.bool()

        span_preds = (span_preds
                    & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                    & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
        match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                            & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
        span_preds = match_label_mask & span_preds

        tp_preds = (span_labels & span_preds).long()
        fp_preds = (~span_labels & span_preds).long()
        fn_preds = (span_labels & ~span_preds).long()

        for idx in range(bsz):
            ent_type = entity_types[idx]
            category[ent_type]["TP"] += tp_preds[idx].sum().item()
            category[ent_type]["FP"] += fp_preds[idx].sum().item()
            category[ent_type]["FN"] += fn_preds[idx].sum().item()
        
        return category
    
    def _compute_category_metrics(self, category):

        for ner_type in category:
            TP = category[ner_type]["TP"]
            FP = category[ner_type]["FP"]
            FN = category[ner_type]["FN"]
            category[ner_type]["support"] = TP + FN

            precision = 0
            if TP + FP != 0:
                precision = TP / (TP + FP)
            
            recall = 0
            if TP + FN != 0:
                recall = TP / (TP + FN)
            
            f1 = 0
            if precision + recall != 0:
                f1 = (2 * precision * recall) / (precision + recall)
            
            category[ner_type]["precision"] = precision
            category[ner_type]["recall"] = recall
            category[ner_type]["f1"] = f1
        
        return category
    
    def _compute_overall_metrics(self, category):
        TP = 0
        FP = 0
        FN = 0
        
        macro_precision = 0
        macro_recall = 0
        macro_f1 = 0

        weighted_precision = 0
        weighted_recall = 0
        weighted_f1 = 0

        for ner_type in category:
            TP += category[ner_type]["TP"]
            FP += category[ner_type]["FP"]
            FN += category[ner_type]["FN"]
            
            macro_precision += category[ner_type]["precision"]
            macro_recall += category[ner_type]["recall"]
            macro_f1 += category[ner_type]["f1"]
            
            num = category[ner_type]["TP"] + category[ner_type]["FN"]
            weighted_precision += category[ner_type]["precision"] * num
            weighted_recall += category[ner_type]["recall"] * num
            weighted_f1 += category[ner_type]["f1"] * num
        
        macro_precision /= len(category)
        macro_recall /= len(category)
        macro_f1 /= len(category)

        support = TP + FN
        if support != 0:
            weighted_precision /= support
            weighted_recall /= support
            weighted_f1 /= support

        micro_precision = 0
        if TP + FP != 0:
            micro_precision = TP / (TP + FP)
        
        micro_recall = 0
        if TP + FN != 0:
            micro_recall = TP / (TP + FN)
        
        micro_f1 = 0
        if micro_precision + micro_recall != 0:
            micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

        overall = {
            "micro-precision": micro_precision,
            "micro-recall": micro_recall,
            "micro-f1": micro_f1,
            "macro-precision": macro_precision,
            "macro-recall": macro_recall,
            "macro-f1": macro_f1,
            "weighted-precision": weighted_precision,
            "weighted-recall": weighted_recall,
            "weighted-f1": weighted_f1,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "support": support,
            "num_pred": TP + FP,
        }

        return overall