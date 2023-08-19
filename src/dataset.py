import torch
from torch.utils.data import Dataset


class MRCNERDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, item_list):
        query_list = [x["query"] for x in item_list]
        context_list = [x["context"] for x in item_list]

        tokenized_text = self.tokenizer(
            query_list,
            context_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        batch_size = len(item_list)
        seq_len = len(tokenized_text.input_ids[0])

        batch_start_label_mask = []
        batch_end_label_mask = []
        batch_start_labels = []
        batch_end_labels = []
        batch_span_labels = []
        entity_types = []

        for idx in range(batch_size):
            item = item_list[idx]
            context = item["context"]
            words = context.split()
            start_word_idxs = item["start_position"]
            end_word_idxs = item["end_position"]
            entity_type = item["entity_label"]
            start_char_idxs = [x + sum([len(w) for w in words[:x]]) for x in start_word_idxs]
            end_char_idxs = [x + sum([len(w) for w in words[:x + 1]]) for x in end_word_idxs]
            assert len(start_char_idxs) == len(end_char_idxs)

            token_type_ids = tokenized_text.token_type_ids[idx]
            offset_mapping = tokenized_text.offset_mapping[idx]

            context_char2token = {}
            for token_idx in range(seq_len):
                if token_type_ids[token_idx] == 0:
                    continue
                start_char, end_char = offset_mapping[token_idx]
                if start_char == end_char == 0:
                    continue
                context_char2token[start_char] = token_idx
                context_char2token[end_char] = token_idx
                
            start_token_idxs = [context_char2token[char_idx] for char_idx in start_char_idxs]
            end_token_idxs = [context_char2token[char_idx] for char_idx in end_char_idxs]
            
            label_mask = []
            for token_idx in range(seq_len):
                if (token_type_ids[token_idx] == 0) or \
                    (offset_mapping[token_idx] == (0, 0)):
                    label_mask.append(0)
                else:
                    label_mask.append(1)
            
            start_label_mask = label_mask.copy()
            end_label_mask = label_mask.copy()

            # start_label_mask: 如果一个word被切分为多个token, 则第一个token的mask为1, 其他token的mask为0; 如果没被切分则mask为1
            # end_label_mask: 如果一个word被切分为多个token, 则最后一个token的mask1, 其他token的mask为0; 如果没被切分则mask为1
            
            word_ids = tokenized_text.word_ids(idx)
            for token_idx in range(seq_len):
                curr_word_id = word_ids[token_idx]
                next_word_id = word_ids[token_idx + 1] if token_idx + 1 < seq_len else None
                prev_word_id = word_ids[token_idx - 1] if token_idx - 1 >= 0 else None
                if (prev_word_id is not None) and (curr_word_id == prev_word_id):
                    start_label_mask[token_idx] = 0
                if (next_word_id is not None) and (curr_word_id == next_word_id):
                    end_label_mask[token_idx] = 0
            
            assert all(start_label_mask[token_idx] != 0 for token_idx in start_token_idxs)
            assert all(end_label_mask[token_idx] != 0 for token_idx in end_token_idxs)

            start_labels = [(1 if token_idx in start_token_idxs else 0) for token_idx in range(seq_len)]
            end_labels = [(1 if token_idx in end_token_idxs else 0) for token_idx in range(seq_len)]

            span_labels = torch.zeros((seq_len, seq_len), dtype=torch.long)
            for start, end in zip(start_token_idxs, end_token_idxs):
                span_labels[start, end] = 1
            
            batch_start_label_mask.append(start_label_mask)
            batch_end_label_mask.append(end_label_mask)
            batch_start_labels.append(start_labels)
            batch_end_labels.append(end_labels)
            batch_span_labels.append(span_labels)
            entity_types.append(entity_type)

        batch = {
            "input_ids": torch.LongTensor(tokenized_text.input_ids),
            "attention_mask": torch.ByteTensor(tokenized_text.attention_mask),
            "token_type_ids": torch.LongTensor(tokenized_text.token_type_ids),
            "start_labels": torch.LongTensor(batch_start_labels),
            "end_labels": torch.LongTensor(batch_end_labels),
            "span_labels": torch.stack(batch_span_labels, dim=0),
            "start_label_mask": torch.ByteTensor(batch_start_label_mask),
            "end_label_mask": torch.ByteTensor(batch_end_label_mask),
            "entity_types": entity_types,
        }

        return batch


def test():
    import os, sys
    import json
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    os.chdir(sys.path[0])

    data = json.load(open("../data/MRC/conll03/mrc-ner.train", encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained("../../pretrained_model/bert-base-uncased/")
    max_length = 512

    dataset = MRCNERDataset(data, tokenizer, max_length)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    for batch in tqdm(dataloader, total=len(dataloader)):
        print(batch.keys())


if __name__ == "__main__":
    test()
            






            


            