import os
os.environ['HF_HOME'] = 'huggingface'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = 'True'
import math
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC, TrainingArguments, Trainer
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
import numpy as np
import evaluate

model_name = 'facebook/wav2vec2-conformer-rel-pos-large-960h-ft'
checkpoint_name = 'checkpoints/checkpoint-750/'

processor = Wav2Vec2Processor.from_pretrained(model_name)

ds = load_dataset('audiofolder', data_dir='imda_truncated_edited', split='train')
ds = ds.train_test_split(test_size=0.2, shuffle=True)

def prepare_dataset(batch):
    model_name = 'facebook/wav2vec2-conformer-rel-pos-large-960h-ft'

    # separate import for each process
    from transformers import Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    batch["input_values"] = [processor(audio["array"], sampling_rate=16000).input_values for audio in batch["audio"]]
    batch["input_length"] = [len(b[0]) for b in batch["input_values"]]
    batch['length'] = batch["input_length"]
    batch["labels"] = processor(text=batch["annotation"]).input_ids

    return batch

ds = ds.map(prepare_dataset, num_proc=8, batched=True, batch_size=256)

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

wer = evaluate.load("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    return {"wer": wer.compute(predictions=pred_str, references=label_str)}

model = Wav2Vec2ConformerForCTC.from_pretrained(
    model_name,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    mask_time_prob=0.7,  # 0.05
    mask_time_length=10, # 10
    mask_feature_prob=0.7, # 0
    mask_feature_length=10, # 10
)

per_gpu_bs = 4
effective_bs = 384
training_args = TrainingArguments(
    output_dir="wav2vec2-checkpoints-imda-bs=384-lr=1e-5",
    overwrite_output_dir =True,
    per_device_train_batch_size=per_gpu_bs,
    gradient_accumulation_steps=math.ceil(effective_bs/per_gpu_bs),
    learning_rate=1e-5,
    num_train_epochs=60,
    gradient_checkpointing=False,
    fp16=True,
    # bf16=True,  # for A100
    fp16_full_eval=True,
    # bf16_full_eval=True,  # for A100
    group_by_length=True,  # slows down
    evaluation_strategy="epoch",
    save_strategy='epoch',  # epoch
    save_safetensors=True,
    per_device_eval_batch_size=4,
    save_steps=1,
    eval_steps=1,
    logging_steps=1,
    save_total_limit=3,
    lr_scheduler_type='cosine',
    load_best_model_at_end=True,  # True
    adam_beta1=0.9,
    adam_beta2=0.98,  # follow fairseq fintuning config

    warmup_ratio=0., # check whether IMDA data works first

#    warmup_ratio=0.22, # follow Ranger21
    weight_decay=1e-4,  # follow Ranger21
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to=['tensorboard'],
    dataloader_num_workers=8 if os.name != 'nt' else 1) # since num threads is 16

class CTCTrainer(Trainer):
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if os.name != 'nt':
            accelerator.backward(self.scaler.scale(loss))
            # self.scaler.scale(loss).backward()
        else:
            self.scaler.scale(loss).backward()
        return loss.detach()
    
if os.name != 'nt':
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision='fp16', dynamo_backend='eager')  # FP8 needs transformer_engine package which is only on Linux with Hopper GPUs

# max_steps = math.ceil(training_args.num_train_epochs * len(ds['train']) / training_args.gradient_accumulation_steps / min(training_args.per_device_train_batch_size, len(ds['train'])))
# optimizer = Ranger21(model.parameters(), num_iterations=max_steps, lr=1e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-8, foreach=False)  # https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/config/finetuning/base_960h.yaml
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=tri_stage_schedule)  # following FAIR finetuning settings
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: x)  # constant LR, stays same throughout, for Ranger21

trainer = CTCTrainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # optimizers=(optimizer, scheduler),
)
if os.name != 'nt':  # windows does not support torch.compile yet
    # pass
    trainer.model_wrapped, trainer.optimizer, trainer.lr_scheduler = accelerator.prepare(trainer.model_wrapped, trainer.optimizer, trainer.lr_scheduler)
trainer.train()
if os.name != 'nt':
    accelerator.wait_for_everyone()
