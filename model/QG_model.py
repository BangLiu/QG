import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from .modules.maxout import MaxOut
from .modules.rnn import StackedGRU
from .modules.attention import ConcatAttention
from .modules.treelstm_utils import st_gumbel_softmax, st_gumbel_softmax2
from .QG_emb import Embedder, ELMoEmbedding
from .QG_clue_qanet import CluePredictor_qanet
from .QG_clue_gcn import CluePredictor_gcn
from .QG_beam import Beam
from .config import *
from util.tensor_utils import to_sorted_tensor, to_original_tensor
from util.tensor_utils import transform_tensor_by_dict
from util.tensor_utils import to_thinnest_padded_tensor
from util.prepro_utils import tokens2ELMOids


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, config, input_size, dropout=0.1):
        super(Encoder, self).__init__()
        self.config = config
        self.layers = config.layers
        self.num_directions = 2 if config.brnn else 1
        assert config.enc_rnn_size % self.num_directions == 0
        self.hidden_size = config.enc_rnn_size // self.num_directions
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=config.layers,
                          dropout=config.dropout,
                          bidirectional=config.brnn,
                          batch_first=True)

    def forward(self, input_emb, lengths, hidden=None):
        # input_emb shape: [seq_len, batch_size, hidden_dim] [100, 32, 412]
        # sorted_emb shape: [seq_len, batch_size, hidden_dim] [100, 32, 412]
        sorted_input_emb, sorted_lengths, sorted_idx = to_sorted_tensor(
            input_emb, lengths, sort_dim=1, device=device)
        emb = pack(sorted_input_emb, sorted_lengths, batch_first=False)
        self.rnn.flatten_parameters()
        outputs, hidden_t = self.rnn(emb, hidden)
        # hidden_t shape:  [num_layers, batch_size, hidden_dim] [2, 32, 256]
        # outputs shape: [unpadded_seq_len, batch_size,
        # hidden_dim * num_layers] [79, 32, 512]
        # !!! NOTICE: it will unpack to max_unpadded_length.
        outputs = unpack(outputs, batch_first=False)[0]
        outputs = to_original_tensor(
            outputs, sorted_idx, sort_dim=1, device=device)
        return hidden_t, outputs


class Decoder(nn.Module):
    def __init__(self, config, dec_input_size, dropout=0.1):
        super(Decoder, self).__init__()
        self.config = config
        self.layers = config.layers
        self.input_feed = config.input_feed
        input_size = dec_input_size
        if self.input_feed:
            input_size += config.enc_rnn_size
        # why use this, not default multi-layer GRU?
        self.rnn = StackedGRU(
            config.layers, input_size, config.dec_rnn_size, config.dropout)
        self.attn = ConcatAttention(
            config.enc_rnn_size, config.dec_rnn_size, config.att_vec_size)
        self.dropout = nn.Dropout(config.dropout)
        self.readout = nn.Linear(
            (config.enc_rnn_size + config.dec_rnn_size + dec_input_size),
            config.dec_rnn_size)
        self.maxout = MaxOut(config.maxout_pool_size)
        self.maxout_pool_size = config.maxout_pool_size
        self.copySwitch = nn.Linear(
            config.enc_rnn_size + config.dec_rnn_size, 1)
        self.hidden_size = config.dec_rnn_size

    def forward(self, output_emb, hidden, context, src_pad_mask, init_att):
        # decoder output_emb shape:
        # [maximum_batch_seq_len, batch_size, emb_dim] [22, 32, 300]
        # decoder hidden shape:
        # [num_layers, batch_size, hidden_dim] [1, 32, 512]
        # decoder context shape:
        # [max_batch_seq_len, batch_size, hidden_dim][79, 32, 512]
        # decoder src_pad_mask shape:
        # [batch_size, max_batch_seq_len][32, 79]
        # decoder init_att shape:
        # [batch_size, hidden_dim] [32, 512]
        g_outputs = []
        c_outputs = []
        copyGateOutputs = []
        cur_context = init_att
        self.attn.applyMask(src_pad_mask)
        precompute = None

        for emb_t in output_emb.split(1):
            # previous output word embedding.
            # tgt start with <SOS>, so we can use t rather than t - 1.
            emb_t = emb_t.squeeze(0)
            # decoder emb_t shape: [batch_size, emb_dim] [32, 300]
            input_emb = emb_t
            # concatenate previous word embedding with previous
            # attention context, use them as decoder rnn input
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_context], 1)

            # get output and new hidden state by decoder rnn
            output, hidden = self.rnn(input_emb, hidden)  # !!!!!!!!
            # decoder rnn output shape: [batch_size, hidden_dim] [32, 512]
            # decoder rnn hidden shape:
            # [num_layers, batch_size, hidden_dim] [1, 32, 512]

            # update new attention context
            cur_context, attn, precompute = self.attn(
                output, context.transpose(0, 1), precompute)

            # use current decoder output + attention context to get copy prob
            copyProb = self.copySwitch(torch.cat((output, cur_context), dim=1))
            copyProb = torch.sigmoid(copyProb)

            # use previous word emb + current rnn output +
            # current attention context, to get final generate output vector
            # which will be fed into generator to generate current output
            # word probabilities
            readout = self.readout(
                torch.cat((emb_t, output, cur_context), dim=1))
            # decoder readout shape:  [batch_size, hidden_dim] [32, 512]
            maxout = self.maxout(readout)
            # decoder maxout shape:
            # [batch_size, hidden_dim / max_out_size] [32, 256]
            output = self.dropout(maxout)
            # decoder output shape at this step:
            # [batch_size, hidden_dim / max_out_size] [32, 256]

            g_outputs += [output]
            # use the attention between current rnn output and context
            # as copy probabilities of context words
            c_outputs += [attn]
            copyGateOutputs += [copyProb]
        g_outputs = torch.stack(g_outputs)
        c_outputs = torch.stack(c_outputs)
        copyGateOutputs = torch.stack(copyGateOutputs)
        # decoder g_outputs shape:
        # [max_batch_output_seq_len, batch_size, hidden_dim / max_out_size]
        # [22, 32, 256]
        # decoder c_outputs shape:
        # [max_batch_output_seq_len, batch_size, max_batch_input_seq_len]
        # [22, 32, 79]
        # decoder copyGateOutputs shape:
        # [max_batch_output_seq_len, batch_size, 1] [22, 32, 1]
        return g_outputs, c_outputs, copyGateOutputs, hidden, attn, cur_context


class DecInit(nn.Module):
    """
    Use encoder's last backward hidden state as input,
    project to decoder rnn size by linear + tanh layers.
    Notice: last backward hidden state remembers more clear about
    head input words, which are more import to head output words.
    If we use single layer RNN for encoder, usually the input word order
    is also reverse due to this reason.
    """
    def __init__(self, config):
        super(DecInit, self).__init__()
        self.num_directions = 2 if config.brnn else 1
        assert config.enc_rnn_size % self.num_directions == 0
        self.enc_rnn_size = config.enc_rnn_size
        self.dec_rnn_size = config.dec_rnn_size
        # initialize decoder hidden state
        input_dim = self.enc_rnn_size // self.num_directions
        self.initer = nn.Linear(
            input_dim,
            self.dec_rnn_size)
        self.tanh = nn.Tanh()

    def forward(self, enc_list):
        x = torch.cat(enc_list, dim=1)
        return self.tanh(self.initer(x))


class Generator(nn.Module):
    """
    Use decoder's final output vector to generate output word
    probabilities.
    """
    def __init__(self, input_size, predict_size):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size, predict_size),
            nn.Softmax(dim=1))

    def forward(self, g_output_t):
        return self.generator(g_output_t)


class QGModel_S2S_CluePredict(nn.Module):
    def __init__(self, config, emb_mats, emb_dicts, dropout=0.1):
        super().__init__()
        self.config = config
        self.config.n_best = 1
        self.dicts = emb_dicts
        self.PAD = emb_dicts["word"]["<PAD>"]

        # input, output embedder
        self.enc_embedder = Embedder(
            config, emb_mats, emb_dicts, dropout)
        if config.share_embedder:
            self.dec_embedder = self.enc_embedder
        else:
            self.dec_embedder = Embedder(
                config, emb_mats, emb_dicts, dropout)
        self.enc_emb_tags = config.emb_tags
        self.dec_emb_tags = ["word"]

        self.src_vocab_limit = config.emb_config["word"]["emb_size"]
        if config.use_refine_copy_tgt_src or config.use_refine_copy_src:
            self.src_vocab_limit = config.refined_src_vocab_limit

        total_emb_size = self.enc_embedder.get_total_emb_dim(self.enc_emb_tags)

        # ELMo embedder
        if config.add_elmo:
            self.elmo_outdim = 1024
            self.elmo_embedder = ELMoEmbedding(config)
            total_emb_size += self.elmo_outdim

        # word freq embedder
        if config.add_word_freq_emb:
            # 0: PAD, 1: low freq, 2: mid freq, 3: high freq
            self.word_freq_embedder = nn.Embedding(
                num_embeddings=4,
                embedding_dim=config.word_freq_emb_dim,
                padding_idx=0)
            total_emb_size += config.word_freq_emb_dim

        # clue predictor
        if self.config.use_clue_predict:
            if self.config.clue_predictor == "qanet":
                self.clue_predictor = CluePredictor_qanet(
                    config, total_emb_size, PAD=self.PAD, dropout=dropout)
            else:
                self.clue_predictor = CluePredictor_gcn(
                    config, total_emb_size, PAD=self.PAD, dropout=dropout)
        if self.config.use_clue_predict:
            self.clue_threshold = 0.5
            clue_embedding_dim = config.emb_config["is_overlap"]["emb_dim"]
            self.clue_embedder = nn.Embedding(
                num_embeddings=3,  # 0: PAD, 1: not overlap, 2: overlap
                embedding_dim=clue_embedding_dim,
                padding_idx=0)

        # clue mask embedding vector
        if self.config.use_clue_mask:
            self.clue_mask_emb = nn.Parameter(
                torch.randn(config.emb_config["word"]["emb_dim"]))
            self.clue_emb_dim = config.emb_config["word"]["emb_dim"]

        # encoder
        enc_input_size = total_emb_size
        if self.config.use_clue_predict:
            enc_input_size += clue_embedding_dim
        self.encoder = Encoder(config, enc_input_size, dropout)

        # decoder
        dec_input_size = config.emb_config["word"]["emb_dim"]
        if config.add_elmo:
            dec_input_size += self.elmo_outdim
        self.decoder = Decoder(config, dec_input_size, dropout)
        self.decIniter = DecInit(config)

        # generator
        self.predict_size = min(config.tgt_vocab_limit, len(emb_dicts["tgt"]))
        if config.use_refine_copy_tgt or config.use_refine_copy_tgt_src:
            self.predict_size = min(
                config.refined_tgt_vocab_limit, len(emb_dicts["tgt"]))
        self.generator = Generator(
            config.dec_rnn_size // config.maxout_pool_size,
            self.predict_size)

    def make_init_att(self, context):
        """
        Create init context attention as zero tensor
        """
        batch_size = context.size(1)  # !!!
        h_size = (
            batch_size,
            self.encoder.hidden_size * self.encoder.num_directions)
        result = context.data.new(*h_size).zero_()
        result.requires_grad = False
        return result

    def refine_clue_predict(self, y_clue, batch):
        """
        Utilize the mid frequency y_clue results,
        keep low and high freq to be 1 and 0.
        """
        word_ids = batch["ans_sent_word_ids"]
        low_freq_bound = self.config.low_freq_bound
        high_freq_bound = self.config.high_freq_bound
        OOV_id = self.dicts["word"]["<OOV>"]
        PAD_id = self.dicts["word"]["<PAD>"]
        low_freq_mask = (word_ids > low_freq_bound).float() + \
                        (word_ids == OOV_id).float()
        high_freq_mask = (word_ids != PAD_id).float() * \
                         (word_ids != OOV_id).float() * \
                         (word_ids < high_freq_bound).float()
        y_clue = ((y_clue + low_freq_mask) > 0).float() * (1 - high_freq_mask)
        return y_clue

    def forward(self, batch):
        src, src_max_len = to_thinnest_padded_tensor(
            batch["ans_sent_word_ids"])
        src_pad_mask = src.data.eq(self.PAD).float()
        src_pad_mask.requires_grad = False
        src = src.transpose(0, 1)

        tgt, tgt_max_len = to_thinnest_padded_tensor(batch["tgt"])
        tgt = tgt.transpose(0, 1)[:-1]  # exclude last <EOS> target from inputs
        if self.config.add_elmo:
            tgt_elmo_ids = batch["ques_elmo_ids"][:, :tgt_max_len - 1, :]

        # input sentence lengths
        maskS = (torch.ones_like(batch["ans_sent_word_ids"]) *
                 self.PAD != batch["ans_sent_word_ids"]).float()
        lengths = maskS.sum(dim=1)

        # embedding input
        input_emb = self.enc_embedder(
            batch, "ans_sent", self.enc_emb_tags).transpose(1, 2)
        # input_emb shape: batch_size * padded_seq_len * hidden_dim

        # add ELMo embedding
        if self.config.add_elmo:
            elmo_emb = self.elmo_embedder(
                batch["ans_sent_elmo_ids"])
            input_emb = torch.cat([input_emb, elmo_emb], dim=2)

        # add word freq embedding
        if self.config.add_word_freq_emb:
            freq_emb = self.word_freq_embedder(
                batch["ans_sent_word_freq"])
            input_emb = torch.cat([input_emb, freq_emb], dim=2)

        # add clue embedding
        y_clue_logits = None
        if self.config.use_clue_predict:
            y_clue_logits, maskA = self.clue_predictor(
                batch, input_emb.transpose(1, 2))
            y_clue = st_gumbel_softmax2(
                y_clue_logits, device=input_emb.get_device())
            if self.config.use_refine_clue:
                y_clue = self.refine_clue_predict(y_clue, batch)

            clue_ids = ((y_clue.float() + 1) * maskS).long()
            clue_emb = self.clue_embedder(clue_ids)
            input_emb = torch.cat([input_emb, clue_emb], dim=2)
            if self.config.use_clue_mask:
                clue_mask = (y_clue.float() * maskS).long()
                for ii in range(input_emb.shape[0]):
                    for jj in range(input_emb.shape[1]):
                        if clue_mask[ii, jj] == 1:
                            input_emb[ii, jj, :self.clue_emb_dim] = self.clue_mask_emb

        # encoding
        input_emb = input_emb.transpose(0, 1)
        # input_emb shape: seq_len * batch_size * hidden_dim
        enc_hidden, context = self.encoder(input_emb, lengths)

        # decoding
        init_att = self.make_init_att(context)
        # [1] is the last backward hiden, NOTICE: it requires must be BRNN
        dec_init_input = [enc_hidden[1]]
        init_dec_hidden = self.decIniter(dec_init_input).unsqueeze(0)

        # !!! as we only feed output word embedding to decoder
        output_emb = self.dec_embedder.embs["word"](tgt)
        if self.config.add_elmo:
            output_elmo_emb = self.elmo_embedder(tgt_elmo_ids).transpose(0, 1)
            output_emb = torch.cat([output_emb, output_elmo_emb], dim=2)
        (g_out, c_out, c_gate_out,
         dec_hidden, _attn, _attention_vector) = self.decoder(
             output_emb, init_dec_hidden, context, src_pad_mask, init_att)

        batch_size = g_out.size(1)  # !!!
        g_out_t = g_out.view(-1, g_out.size(2))
        g_prob_t = self.generator(g_out_t)
        g_prob_t = g_prob_t.view(-1, batch_size, g_prob_t.size(1))
        # g_prob_t shape: [max_batch_seq_len, batch_size, predict_size]

        return g_prob_t, c_out, c_gate_out, y_clue_logits, src_max_len

    def buildTargetTokens(self, pred, src, isCopy, copyPosition, attn):
        pred_word_ids = [x.item() for x in pred]
        tokens = []
        # generate
        for i in pred_word_ids:
            if self.config.use_generated_tgt_as_tgt_vocab:
                tokens.append(self.dicts["idx2generated_tgt"].get(i))
                if i == self.dicts["generated_tgt"]["<EOS>"]:
                    break
            else:
                tokens.append(self.dicts["idx2tgt"].get(i))
                if i == self.dicts["tgt"]["<EOS>"]:
                    break
        tokens = tokens[:-1]  # delete EOS
        # copy
        for i in range(len(tokens)):
            if isCopy[i]:
                tokens[i] = '[[{0}]]'.format(
                    src[copyPosition[i] - self.predict_size])
        # replace unknown words
        for i in range(len(tokens)):
            if tokens[i] == "<OOV>":
                _, maxIndex = attn[i].max(0)
                tokens[i] = src[maxIndex[0]].encode("utf8").decode("utf8")
        return tokens

    def translate_batch(self, batch):
        src, src_max_len = to_thinnest_padded_tensor(
            batch["ans_sent_word_ids"])
        src = src.transpose(0, 1)
        batchSize = src.size(1)
        beamSize = self.config.beam_size

        #  (1) run the encoder on the src
        # input sentence lengths
        maskS = (torch.ones_like(batch["ans_sent_word_ids"]) *
                 self.PAD != batch["ans_sent_word_ids"]).float()
        lengths = maskS.sum(dim=1)

        input_emb = self.enc_embedder(
            batch, "ans_sent", self.enc_emb_tags).transpose(1, 2)

        # add ELMo embedding
        if self.config.add_elmo:
            elmo_emb = self.elmo_embedder(
                batch["ans_sent_elmo_ids"])
            input_emb = torch.cat([input_emb, elmo_emb], dim=2)

        # add word freq embedding
        if self.config.add_word_freq_emb:
            freq_emb = self.word_freq_embedder(
                batch["ans_sent_word_freq"])
            input_emb = torch.cat([input_emb, freq_emb], dim=2)

        # add clue embedding
        y_clue = None
        if self.config.use_clue_predict:
            y_clue_logits, maskA = self.clue_predictor(
                batch, input_emb.transpose(1, 2))
            y_clue = st_gumbel_softmax2(
                y_clue_logits, device=input_emb.get_device())
            if self.config.use_refine_clue:
                y_clue = self.refine_clue_predict(y_clue, batch)

            clue_ids = ((y_clue.float() + 1) * maskS).long()
            clue_emb = self.clue_embedder(clue_ids)
            input_emb = torch.cat([input_emb, clue_emb], dim=2)

            if self.config.use_clue_mask:
                clue_mask = (y_clue.float() * maskS).long()
                for ii in range(input_emb.shape[0]):
                    for jj in range(input_emb.shape[1]):
                        if clue_mask[ii, jj] == 1:
                            input_emb[ii, jj, :self.clue_emb_dim] = self.clue_mask_emb

        # encoding
        input_emb = input_emb.transpose(0, 1)
        enc_hidden, context = self.encoder(input_emb, lengths)

        dec_init_input = [enc_hidden[1]]
        dec_hidden = self.decIniter(dec_init_input)  # batch, dec_hidden
        # Expand tensors for each beam.
        context = context.data.repeat(1, beamSize, 1)
        dec_hidden = dec_hidden.unsqueeze(0).data.repeat(1, beamSize, 1)
        att_vec = self.make_init_att(context)
        padMask = src.data.eq(self.dicts["word"]["<PAD>"]).transpose(
            0, 1).unsqueeze(0).repeat(beamSize, 1, 1).float()

        beam = [Beam(beamSize) for k in range(batchSize)]
        batchIdx = list(range(batchSize))
        remainingSents = batchSize

        for i in range(self.config.sent_limit):  # !!! sent_limit
            # Prepare decoder input.
            input = torch.stack(
                [b.getCurrentState() for b in beam
                 if not b.done]).transpose(0, 1).contiguous().view(1, -1)
            # print("input size: ", input.shape)
            # NOTICE:::: transform targets and check here
            if self.config.use_generated_tgt_as_tgt_vocab:
                input = transform_tensor_by_dict(
                    input, self.dicts["generated_tgt2word"],
                    input.get_device()).long()
                input = input * (input < self.src_vocab_limit).long() + \
                    self.dicts["tgt"]["<OOV>"] * \
                    (input >= self.src_vocab_limit).long()

            output_emb = self.dec_embedder.embs["word"](input)

            # add ELMo embedding
            if self.config.add_elmo:
                # turn beam outputs to words
                input_words = [[self.dicts["idx2tgt"][tgt_idx]
                                for tgt_idx in sent]
                               for sent in input.tolist()]
                # turn words to elmo ids
                input_elmo_ids = torch.LongTensor(np.stack(
                    [tokens2ELMOids(token_list, len(token_list))
                     for token_list in input_words])).to(device)
                # get elmo embedding and cat
                output_elmo_emb = self.elmo_embedder(input_elmo_ids)
                output_emb = torch.cat([output_emb, output_elmo_emb], dim=2)

            g_outputs, c_outputs, copyGateOutputs, dec_hidden, attn, att_vec = \
                self.decoder(
                    output_emb, dec_hidden, context,
                    padMask.view(-1, padMask.size(2)), att_vec)  # !!!!  in debug mode, the word emb don't have 20000 unique words, therefore, it will cause index out-of-range error.

            # g_outputs: 1 x (beam*batch) x numWords
            copyGateOutputs = copyGateOutputs.view(-1, 1)
            g_outputs = g_outputs.squeeze(0)
            g_out_prob = self.generator.forward(g_outputs) + 1e-8
            g_predict = torch.log(
                g_out_prob * ((1 - copyGateOutputs).expand_as(g_out_prob)))
            c_outputs = c_outputs.squeeze(0) + 1e-8
            c_predict = torch.log(
                c_outputs * (copyGateOutputs.expand_as(c_outputs)))

            # batch x beam x numWords
            wordLk = g_predict.view(
                beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            copyLk = c_predict.view(
                beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(
                beamSize, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            father_idx = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(
                        wordLk.data[idx], copyLk.data[idx], attn.data[idx]):
                    active += [b]
                    father_idx.append(beam[b].prevKs[-1])  # this is very annoying

            if not active:
                break

            # to get the real father index
            real_father_idx = []
            for kk, idx in enumerate(father_idx):
                real_father_idx.append(idx * len(father_idx) + kk)

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = torch.LongTensor(
                [batchIdx[k] for k in active]).to(device)
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t, rnnSize):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return view.index_select(1, activeIdx).view(*newSize)

            dec_hidden = updateActive(dec_hidden, self.config.dec_rnn_size)
            context = updateActive(context, self.config.enc_rnn_size)
            att_vec = updateActive(att_vec, self.config.enc_rnn_size)
            padMask = padMask.index_select(1, activeIdx)

            # set correct state for beam search
            previous_index = torch.stack(real_father_idx).transpose(
                0, 1).contiguous()
            dec_hidden = dec_hidden.view(-1, dec_hidden.size(2)).index_select(
                0, previous_index.view(-1)).view(*dec_hidden.size())
            att_vec = att_vec.view(-1, att_vec.size(1)).index_select(
                0, previous_index.view(-1)).view(*att_vec.size())

            remainingSents = len(active)

        allHyp, allScores, allAttn = [], [], []
        allIsCopy, allCopyPosition = [], []
        n_best = self.config.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            valid_attn = src.data[:, b].ne(
                self.dicts["word"]["<PAD>"]).nonzero().squeeze(1)
            hyps, isCopy, copyPosition, attn = zip(
                *[beam[b].getHyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            allHyp += [hyps]
            allAttn += [attn]
            allIsCopy += [isCopy]
            allCopyPosition += [copyPosition]

        predBatch = []
        src_batch = batch["src_tokens"]
        for b in range(batchSize):
            n = 0
            predBatch.append(
                self.buildTargetTokens(
                    allHyp[b][n], src_batch[b], allIsCopy[b][n],
                    allCopyPosition[b][n], allAttn[b][n])
            )
        return predBatch, y_clue, maskS
