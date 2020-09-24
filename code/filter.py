# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020 <Huawei TAIE/Haluk Açarçiçek>
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import numpy as np
import torch
from override import NMTTXLMRobertaForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import (
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_processors as processors
from transformers.data.processors.utils import InputExample

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, NMTTXLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}


def list2file(arr, path):
    with open(path, 'w') as f:
        for item in arr:
            f.write("%s\n" % str(item).strip())


def file2list(file_name, as_float=False):
    with open(file_name) as f:
        line_list = f.readlines()
    if as_float:
        return [float(line.strip()) for line in line_list]
    else:
        return [line.strip() for line in line_list]


def get_test_examples(args):
    """See base class."""

    src = file2list(args.src_data)
    trg = file2list(args.trg_data)

    return _create_examples(src, trg, "test")


def _create_examples(src_sents, trg_sents, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, (s, t)) in enumerate(zip(src_sents, trg_sents)):
        guid = "%s-%s" % (set_type, str(i))
        text_a = s
        text_b = t
        label = "0"
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def load_and_cache_examples(task, tokenizer, args):
    # torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    output_mode = "classification"

    label_list = ["0", "1"]
    if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]

    examples = (
        get_test_examples(args)
    )

    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode=output_mode,
        pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def evaluate(model, tokenizer, prefix, args):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.model_checkpoint_path,)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(eval_task, tokenizer, args)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        eval_batch_size = args.batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # Eval!
        print("***** Running evaluation {} *****".format(prefix))
        print("  Num examples = %d", len(eval_dataset))
        print("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to("cuda") for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = torch.nn.functional.softmax(logits).detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, torch.nn.functional.softmax(logits).detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = preds
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=1024,
        type=int,
        help="per_gpu_eval_batch_size",
    )
    parser.add_argument(
        "--src_data",
        default=None,
        type=str,
        required=True,
        help="The src sents file. ",
    )

    parser.add_argument(
        "--trg_data",
        default=None,
        type=str,
        required=True,
        help="The trg sents file. ",
    )

    parser.add_argument(
        "--model_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="model_checkpoint_path. ",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="output_dir. ",
    )

    parser.add_argument(
        "--model_type",
        default="xlmroberta",
        type=str,
        help="model_type. ",
    )
    parser.add_argument(
        "--task_name",
        default="sts-b",
        type=str,
        help="task_name. ",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="xlm-roberta-large",
        type=str,
        help="model_name_or_path. ",
    )
    parser.add_argument(
        "--output_mode",
        default="classification",
        type=str,
        help="output_mode. ",
    )
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Evaluation
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint_path, do_lower_case=False)
    checkpoints = [args.model_checkpoint_path]

    print("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

        model = model_class.from_pretrained(checkpoint)
        model.to("cuda")
        preds = evaluate(model, tokenizer, prefix, args)

    list2file(preds[:, 1], args.output_dir + "scores.txt")


if __name__ == "__main__":
    main()
