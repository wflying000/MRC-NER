import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        loss_computer,
        metrics_computer,
        training_args,
        output_dir,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_computer = loss_computer
        self.metrics_computer = metrics_computer
        self.training_args = training_args
        self.output_dir = output_dir
        self.writer = SummaryWriter(output_dir)

        self.device = None
        for n, p in model.named_parameters():
            self.device = p.device

    def train(self):
        model = self.model
        train_dataloader = self.train_dataloader
        optimizer = self.optimizer
        scheduler = self.scheduler
        args = self.training_args
        num_train_batches = len(train_dataloader)

        for epoch in tqdm(range(args.num_train_epochs), total=args.num_train_epochs):
            model.train()

            total_loss = 0
            total_start_loss = 0
            total_end_loss = 0
            total_span_loss = 0

            self.metrics_computer.clear()

            for batch_idx, batch in tqdm(enumerate(train_dataloader), total=num_train_batches, leave=False):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                outputs = model(batch)

                start_logits = outputs["start_logits"]
                end_logits = outputs["end_logits"]
                span_logits = outputs["span_logits"]
                start_labels = batch["start_labels"]
                end_labels = batch["end_labels"]
                span_labels = batch["span_labels"]
                start_label_mask = batch["start_label_mask"]
                end_label_mask = batch["end_label_mask"]

                loss, start_loss, end_loss, span_loss = self.loss_computer(
                    start_logits,
                    end_logits,
                    span_logits,
                    start_labels,
                    end_labels,
                    span_labels,
                    start_label_mask,
                    end_label_mask,
                )

                loss.backward()
                torch.cuda.empty_cache()
                if (batch_idx + 1) % args.grad_accumulation_steps == 0 or (batch_idx + 1) == num_train_batches:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()

                total_loss += loss.item()
                total_start_loss += start_loss.item()
                total_end_loss += end_loss.item()
                total_span_loss += span_loss.item()

                global_step = num_train_batches * epoch + batch_idx
                self.writer.add_scalar("Loss/Step/Train/loss", loss.item(), global_step=global_step)
                self.writer.add_scalar("Loss/Step/Train/start_loss", start_loss.item(), global_step=global_step)
                self.writer.add_scalar("Loss/Step/Train/end_loss", end_loss.item(), global_step=global_step)
                self.writer.add_scalar("Loss/Step/Train/span_loss", span_loss.item(), global_step=global_step)
                self.writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], global_step=global_step)

                self.metrics_computer.add_batch(
                    start_logits.detach().cpu(), 
                    end_logits.detach().cpu(), 
                    span_logits.detach().cpu(), 
                    start_label_mask.detach().cpu(), 
                    end_label_mask.detach().cpu(), 
                    span_labels.detach().cpu(), 
                    batch["entity_types"],
                )

            train_metrics = self.metrics_computer.compute()
            train_loss = total_loss / num_train_batches
            train_start_loss = total_start_loss / num_train_batches
            train_end_loss = total_end_loss / num_train_batches
            train_span_loss = total_span_loss / num_train_batches

            self.writer.add_scalar("Loss/Epoch/Train/loss", train_loss, global_step=epoch)
            self.writer.add_scalar("Loss/Epoch/Train/start_loss", train_start_loss, global_step=epoch)
            self.writer.add_scalar("Loss/Epoch/Train/end_loss", train_end_loss, global_step=epoch)
            self.writer.add_scalar("Loss/Epoch/Train/span_loss", train_span_loss, global_step=epoch)

            train_entity_metrics = train_metrics["category"]
            train_overall_metrics = train_metrics["overall"]
            for ent_type, metrics in train_entity_metrics.items():
                for key, value in metrics.items():
                    self.writer.add_scalar(f"Train-Entity/{ent_type}/{key}", value, global_step=epoch)
            for key, value in train_overall_metrics.items():
                self.writer.add_scalar(f"Train-overall/{key}", value, global_step=epoch)

            eval_outputs = self.evaluate()
            eval_metrics = eval_outputs["metrics"]
            eval_loss = eval_outputs["loss"]
            eval_start_loss = eval_outputs["start_loss"]
            eval_end_loss = eval_outputs["end_loss"]
            eval_span_loss = eval_outputs["span_loss"]
            
            self.writer.add_scalar("Loss/Epoch/Eval/loss", eval_loss, global_step=epoch)
            self.writer.add_scalar("Loss/Epoch/Eval/start_loss", eval_start_loss, global_step=epoch)
            self.writer.add_scalar("Loss/Epoch/Eval/end_loss", eval_end_loss, global_step=epoch)
            self.writer.add_scalar("Loss/Epoch/Eval/span_loss", eval_span_loss, global_step=epoch)

            eval_entity_metrics = eval_metrics["category"]
            eval_overall_metrics = eval_metrics["overall"]
            for ent_type, metrics in eval_entity_metrics.items():
                for key, value in metrics.items():
                    self.writer.add_scalar(f"Eval-Entity/{ent_type}/{key}", value, global_step=epoch)
            for key, value in eval_overall_metrics.items():
                self.writer.add_scalar(f"Eval-overall/{key}", value, global_step=epoch)

    def evaluate(self):
        model = self.model
        model.eval()
        num_eval_batches = len(self.eval_dataloader)

        total_loss = 0
        total_start_loss = 0
        total_end_loss = 0
        total_span_loss = 0

        self.metrics_computer.clear()

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, total=num_eval_batches, leave=False):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                outputs = model(batch)

                start_logits = outputs["start_logits"]
                end_logits = outputs["end_logits"]
                span_logits = outputs["span_logits"]
                start_labels = batch["start_labels"]
                end_labels = batch["end_labels"]
                span_labels = batch["span_labels"]
                start_label_mask = batch["start_label_mask"]
                end_label_mask = batch["end_label_mask"]

                loss, start_loss, end_loss, span_loss = self.loss_computer(
                    start_logits,
                    end_logits,
                    span_logits,
                    start_labels,
                    end_labels,
                    span_labels,
                    start_label_mask,
                    end_label_mask,
                )

                total_loss += loss.item()
                total_start_loss += start_loss.item()
                total_end_loss += end_loss.item()
                total_span_loss += span_loss.item()

                self.metrics_computer.add_batch(
                    start_logits.detach().cpu(), 
                    end_logits.detach().cpu(), 
                    span_logits.detach().cpu(), 
                    start_label_mask.detach().cpu(), 
                    end_label_mask.detach().cpu(), 
                    span_labels.detach().cpu(), 
                    batch["entity_types"],
                )
        
        loss = total_loss / num_eval_batches
        start_loss = total_loss / num_eval_batches
        end_loss = total_end_loss / num_eval_batches
        span_loss = total_span_loss / num_eval_batches

        metrics = self.metrics_computer.compute()

        outputs = {
            "loss": loss,
            "start_loss": start_loss,
            "end_loss": end_loss,
            "span_loss": span_loss,
            "metrics": metrics,
        }

        return outputs







        