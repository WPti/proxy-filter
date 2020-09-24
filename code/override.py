import torch
import torch.nn as nn
import random
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from transformers import XLMRobertaForSequenceClassification
from over_utils import DataProcessor, InputExample, InputFeatures


def file2list(file_name, as_float=False):
    with open(file_name) as f:
        line_list = f.readlines()
    if as_float:
        return [float(l.strip()) for l in line_list]
    else:
        return [l.strip() for l in line_list]


def list2file(l, path):
    with open(path, 'w') as f:
        for item in l:
            f.write("%s\n" % str(item).strip())


def bitext_to_sts(src, trg, pos_ratio=1, rand_ratio= 6, fuzzy_max=60, fuzzy_ratio=0, neigbour_mix=True):
    size = len(src)
    sts = []
    t = {k: v for k, v in enumerate(trg)}
    for i in range(size):
        for j in range(pos_ratio):
            sts.append(src[i].strip() + "\t" + trg[i].strip()+ "\t" + "1.0")
        for k in range(rand_ratio):
            sts.append(src[random.randrange(1,size)].strip() + "\t" + trg[i].strip() + "\t" + "0.0")
        if fuzzy_ratio>0:   
            matches = process.extract(trg[i], t, scorer=fuzz.token_sort_ratio, limit=25)
            m_index = [m[2] for m in matches if m[1]<fuzzy_max][:fuzzy_ratio]
            for m in m_index:
                sts.append(src[i].strip() + "\t" + trg[m].strip() + "\t" + "0.0")
        
        if neigbour_mix and i<size-2:
            sts.append(src[i].strip() + "\t" + trg[i+1].strip()+ "\t" + "0.0")
            sts.append(src[i].strip() + "\t" + trg[i-1].strip()+ "\t" + "0.0")
    return sts 


class NMTTXLMRobertaForSequenceClassification(XLMRobertaForSequenceClassification):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__(config)
        self.classifier = NMTTRobertaClassificationHead(config)


class NMTTRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(1024, 2048)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(2048, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class XSTSProcessor(DataProcessor):
    """Processor for the XSTS data set (Huawei version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def create_train_examples(self, pair_dirs, negative_random_sampling, positive_oversampling, two_way_neighbour_sampling, fuzzy_ratio, fuzzy_max ):
        """Creates examples for the training and dev sets."""
        src_lines = []
        trg_lines = []
        for pair in pair_dirs:
            src_lines += file2list(pair["src"])
            trg_lines += file2list(pair["trg"])
        train_sts = bitext_to_sts(src_lines, trg_lines, rand_ratio= negative_random_sampling, pos_ratio=positive_oversampling, neigbour_mix=two_way_neighbour_sampling, fuzzy_ratio=fuzzy_ratio, fuzzy_max=fuzzy_max)
        examples = []
        for (i, line) in enumerate(train_sts):
            guid = "%s-%s" % ("train", str(i))
            text_a ,text_b, label = line.split("\t")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=str(int(float(label)))))
        return examples

    def create_valid_examples(self, valid_pair):
        """Creates examples for the training and dev sets."""
        src_lines = file2list(valid_pair["src"])
        trg_lines = file2list(valid_pair["trg"])
        valid_sts = bitext_to_sts(src_lines, trg_lines, rand_ratio=0, pos_ratio=1, fuzzy_ratio=3, neigbour_mix=True)
        examples = []
        for (i, line) in enumerate(valid_sts):
            guid = "%s-%s" % ("dev", str(i))
            text_a ,text_b, label = line.split("\t")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=str(int(float(label)))))
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, str(i))
            text_a = line[0]
            text_b = line[1]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=str(int(float(label)))))
        return examples
