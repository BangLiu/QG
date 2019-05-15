import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Embedder(nn.Module):
    """
    Embedding different features according to configuration
    and concatenate all the embeddings.
    """
    def __init__(self, config, emb_mats, emb_dicts, dropout=0.1):
        super().__init__()
        self.config = config
        self.embs = torch.nn.ModuleDict()
        self.conv2ds = torch.nn.ModuleDict()
        # construct all keys, so we reuse one embedder
        # and can train on different tasks
        #for tag in emb_mats.keys():
        for tag in config.emb_tags:
            if config.emb_config[tag]["need_emb"]:
                self.embs.update(
                    {tag:
                     nn.Embedding.from_pretrained(
                         torch.FloatTensor(emb_mats[tag]),
                         freeze=(not config.emb_config[tag]["trainable"]))})
                if config.emb_config[tag]["need_conv"]:
                    self.conv2ds.update(
                        {tag:
                         nn.Conv2d(
                             config.emb_config[tag]["emb_dim"], config.d_model,
                             kernel_size=(1, 5), padding=0, bias=True)})
                    nn.init.kaiming_normal_(
                        self.conv2ds[tag].weight, nonlinearity='relu')

        # self.conv1d = Initialized_Conv1d(
        #     total_emb_dim, config.d_model, bias=False)
        # self.high = Highway(2, config.d_model)
        self.dropout = dropout

    def get_total_emb_dim(self, emb_tags):
        total_emb_dim = 0
        for tag in emb_tags:
            if self.config.emb_config[tag]["need_emb"]:
                if self.config.emb_config[tag]["need_conv"]:
                    total_emb_dim += self.config.d_model
                else:
                    total_emb_dim += self.config.emb_config[tag]["emb_dim"]
            else:
                total_emb_dim += 1  # use feature value itself as embedding
        return total_emb_dim

    def forward(self, batch, field, emb_tags):
        emb = torch.FloatTensor().to(device)
        # NOTICE: use emb_tags to control which tags are actually in use
        for tag in emb_tags:
            # NOTICE: naming style is same with data loader of SQuAD
            field_id = field + "_" + tag + "_ids"
            field_tag = field + "_" + tag
            if self.config.emb_config[tag]["need_emb"]:
                tag_emb = self.embs[tag](batch[field_id])
            else:
                tag_emb = batch[field_tag].unsqueeze(2)
            if self.config.emb_config[tag]["need_conv"]:
                tag_emb = tag_emb.permute(0, 3, 1, 2)
                tag_emb = F.dropout(
                    tag_emb, p=self.dropout, training=self.training)
                tag_emb = self.conv2ds[tag](tag_emb)
                tag_emb = F.relu(tag_emb)
                tag_emb, _ = torch.max(tag_emb, dim=3)
            else:
                tag_emb = F.dropout(
                    tag_emb, p=self.dropout, training=self.training)
                tag_emb = tag_emb.transpose(1, 2)
            emb = torch.cat([emb, tag_emb], dim=1)
        # emb = self.conv1d(emb)
        # emb = self.high(emb)
        return emb


class ELMoEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.elmo_wrapper = Elmo(
            config.elmo_options_file, config.elmo_weight_file, 1,
            dropout=config.elmo_dropout_prob,
            requires_grad=config.elmo_requires_grad,
            do_layer_norm=config.elmo_do_layer_norm)

    def forward(self, batch_ids):
        """
        batch_ids is a batch of elmo ids, shape batch_size * padded_length * 50
        """
        elmo_res = self.elmo_wrapper(batch_ids)
        elmo_embedding = elmo_res['elmo_representations'][0]
        return elmo_embedding
