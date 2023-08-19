import torch
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup, 
    get_polynomial_decay_schedule_with_warmup
)


def configure_optimizers(
    model, 
    train_dataloader, 
    optimizer="torch.adam", 
    weight_decay=0.01, 
    lr=3e-5, 
    lr_mini=3e-7, 
    num_gpus=1, 
    grad_accumulation_steps=4, 
    max_epochs=20, 
    lr_scheduler="polydecay", 
    warmup_steps=0, 
    final_div_factor=1e4
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if optimizer == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  
                          lr=lr)
    elif optimizer == "torch.adam":
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                      lr=lr,
                                      weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    total_steps = (len(train_dataloader) // (grad_accumulation_steps * num_gpus) + 1) * max_epochs

    if lr_scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, pct_start=float(warmup_steps/total_steps),
            final_div_factor=final_div_factor,
            total_steps=total_steps, anneal_strategy='linear'
        )
    elif lr_scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    elif lr_scheduler == "polydecay":
        if lr_mini == -1:
            lr_mini = lr / 5
        else:
            lr_mini = lr_mini
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, total_steps, lr_end=lr_mini)
    else:
        raise ValueError
    
    return optimizer, scheduler