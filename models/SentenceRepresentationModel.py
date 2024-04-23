from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from typing import List, Union, Dict, Optional, Callable
from datasets import Dataset
from transformers.data.data_collator import DataCollatorWithPadding
from utils import (
    prepare_inputs,
    remove_unused_columns,
    get_hidden_states,
    get_pooled_layer_representations,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import inspect
import pprint
import torch
import torch.nn as nn

device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class SentenceRepresentationModel:
    """
    Wrapper for a PyTorch model for the sole purpose of retrieving the sentence
    level representations of each specified layer, given some sample input

    The main utility is to encode a large dataset of inputs into sentence
    level representations
    """

    def __init__(
        self,
        sentence1_key: Optional[str] = "text",
        sentence2_key: Optional[str] = "none",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model: Union[PreTrainedModel, nn.Module] = None,
    ):
        torch.cuda.empty_cache()  # save memory before iterating through dataset
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    def forward(
        self,
        encoded_inputs: List[int],
        database: Dict[int, np.ndarray],
        layers_to_save: List[int] = None,
        poolers: List[Callable[[torch.tensor], torch.tensor]] = None,
    ):
        # convert each list in the dictionary to tensor on the device (if no gpu)
        if not torch.cuda.is_available():
            for k in encoded_inputs:
                encoded_inputs[k] = torch.tensor(encoded_inputs[k]).to(device)
        attention_mask = encoded_inputs["attention_mask"].detach().cpu()
        with torch.no_grad():
            if "t5" in self.model.config.model_type:
                # print(encoded_inputs)
                # batch_size = encoded_inputs['input_ids'].shape[0]
                encoded_inputs["decoder_input_ids"] = (
                    torch.tensor([self.tokenizer.pad_token_id])
                    .repeat(encoded_inputs["labels"].shape)
                    .to(device)
                )
                # encoded_inputs['decoder_input_ids'] = encoded_inputs['input_ids']
                # encoded_inputs['decoder_attention_mask'] = encoded_inputs['attention_mask']
                # outputs = self.model.generate(**encoded_inputs, output_hidden_states=True)
                # print(self.tokenizer.batch_decode(outputs))
                # print(outputs)
                # input()
            del encoded_inputs["labels"]
            outputs = self.model(
                **encoded_inputs,
                output_hidden_states=True,
                output_attentions=True,
            )
        # pprint.pprint(vars(outputs))
        # input()
        # print(len(outputs['decoder_hidden_states']))
        # print(outputs['encoder_hidden_states'][0].shape)
        # print(outputs['decoder_hidden_states'][0].shape)
        # input()
        hidden_states = get_hidden_states(
            self.model.config.is_encoder_decoder, outputs
        )
        # Hidden-states of the model = the initial embedding outputs + the output of each layer
        # filter representations to what we need: (num_layers+1, batch_size, max_seq_len, embedding_dim)
        for layer, pooler in zip(layers_to_save, poolers):
            if self.model.config.is_encoder_decoder:
                if layer < len(outputs["encoder_hidden_states"]):
                    layer_rep_np = pooler(hidden_states[layer], attention_mask)
                else:
                    layer_rep_np = pooler(
                        hidden_states[layer],
                        torch.ones(hidden_states[layer].shape[:2]),
                    )
            else:
                layer_rep_np = pooler(hidden_states[layer], attention_mask)
            # layer_rep_np = np.concatenate(
            #     (layer_rep_np, tags.reshape(-1, 1), labels.reshape(-1, 1)), axis=1) # (batch_size, embedding_dim + 2)
            database[layer] = (
                np.append(database[layer], layer_rep_np, axis=0)
                if database[layer] is not None
                else layer_rep_np
            )
        return database

    def tokenize(self, examples: Union[str, List[str]]):
        args = (
            (examples[self.sentence1_key],)
            if self.sentence2_key == "none"
            else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(
            *args,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        if "label" in examples:
            if "t5" in self.model.config.model_type:
                result["label"] = self.tokenizer(
                    examples["label"],
                    padding=True,
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                ).input_ids
        return result

    def encode_batch(
        self,
        exs: List[str],
        layers_to_save: List[int] = None,
        poolers: List[Callable[[torch.tensor], torch.tensor]] = None,
    ):
        encoded_inputs = self.tokenizer(
            exs,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        database = {layer: None for layer in layers_to_save}
        return self.forward(encoded_inputs, database, layers_to_save, poolers)

    def encode_dataset(
        self,
        data: pd.DataFrame,
        intToText: List[str],
        batch_size: int,
        layers_to_save: List[int] = None,
        poolers: List[Callable[[torch.tensor], torch.tensor]] = None,
    ):
        if "t5" in self.model.config.model_type:
            data["label"] = list(map(lambda x: intToText[x], data["label"]))
        dataset = Dataset.from_pandas(data).map(
            self.tokenize,
            batched=True,
            load_from_cache_file=False,
            desc=f"Running tokenizer on dataset",
        )
        # Inspect model forward signature to keep only the arguments it accepts.
        # Note: do not shuffle examples (preserve original ordering)
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(self.model.forward)
        signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        signature_columns += ["label", "label_ids"]
        dataloader = DataLoader(
            remove_unused_columns(dataset, self.model, True, signature_columns),
            shuffle=False,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
        )
        progress_bar = tqdm(range(len(dataloader)))
        database = {layer: None for layer in layers_to_save}
        for batch in dataloader:
            inputs = prepare_inputs(batch, signature_columns, device)
            database = self.forward(inputs, database, layers_to_save, poolers)
            progress_bar.update(1)
        return database


def get_model_for_representation(
    model_name_or_path,
    num_labels=2,  # binary classification
    ignore_mismatched_sizes=False,
    cache_dir="/work/06782/ysu707/ls6/DkNN/.cache",
    max_seq_length=1024,
    sentence1_key: Optional[str] = "text",
    sentence2_key: Optional[str] = "none",
):
    # Load pretrained model and tokenizer for training
    config = AutoConfig.from_pretrained(
        model_name_or_path, num_labels=num_labels, cache_dir=cache_dir
    )
    # max_position_embeddings=512 by default, which may be less than the maximum sequence length
    # if that's case, then we randomly initialize the classification head by passing
    # ignore_mismatched_sizes = True

    if "t5" in config.model_type:
        ignore_mismatched_sizes = False
    else:
        ignore_mismatched_sizes = True
        if config.max_position_embeddings != max_seq_length:
            config.max_position_embeddings = max_seq_length

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, cache_dir=cache_dir, use_fast=False
    )

    if "t5" in config.model_type:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

    m = SentenceRepresentationModel(
        sentence1_key=sentence1_key,
        sentence2_key=sentence2_key,
        tokenizer=tokenizer,
        model=model.cuda() if torch.cuda.is_available() else model,
    )

    return m
