import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
import dni

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, batchsize, rnn_type, ntoken, ninp, nhid, nlayers, 
        dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, use_sg=False):
        super(RNNModel, self).__init__()

        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM','QRNN', 'GRU', 'INDRNN'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)

        # self.lns = [torch.nn.LayerNorm(nhid if l != nlayers - 1 else (ninp if tie_weights else nhid)) for l in range(nlayers)]
        print(self.rnns)

        self.rnns = torch.nn.ModuleList(self.rnns)
        # self.lns = torch.nn.ModuleList(self.lns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

        self.init_weights()

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, use_sg=False, backward_interface=None):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []

        raw_outputs = []
        outputs = []

        if backward_interface:
            hidden = self.join_hidden(hidden)
            hidden = backward_interface.make_trigger(hidden)
            hidden = self.split_hidden(hidden)

        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])

            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)

        if backward_interface and use_sg and self.training:
            new_hidden = self.join_hidden(new_hidden)
            new_hidden.backward(backward_interface.receive(new_hidden).data * 0.1, retain_graph=True)
            new_hidden = self.split_hidden(new_hidden)
        hidden = new_hidden
        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.reshape(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU' or self.rnn_type == 'INDRNN':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]

    def join_hidden(self, hidden):
        if isinstance(hidden[0], torch.Tensor):
            return torch.cat(hidden, dim=2)
        else:
            return self.join_hidden([self.join_hidden(h) for h in hidden])

    def split_hidden(self, hidden):
        if self.rnn_type == "LSTM":
            hidden = torch.split(hidden, 
                split_size_or_sections=[(self.nhid if l < (self.nlayers - 1)*2 else (self.ninp if self.tie_weights else self.nhid)) for l in range(self.nlayers*2) ], 
                dim=2)
            hidden = [h.contiguous() for h in hidden]
            hidden = [ (hidden[l],hidden[l+1]) for l in range(0,self.nlayers*2,2)]
        else:
            hidden = torch.split(hidden, 
                split_size_or_sections=[self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid) for l in range(self.nlayers) ], 
                dim=2)
            hidden = [h.contiguous() for h in hidden]
        return hidden

class RNNSynthesizer(nn.Module):
    def __init__(self, sgtype="GRU", nhid=50, nlayers=2, input_features=1, sg_dropout=0.):
        super(RNNSynthesizer, self).__init__()
        self.encoder = nn.Linear(in_features = input_features, out_features = nhid)

        if sgtype == "GRU":
            self.rnns = [torch.nn.GRU(nhid, nhid, dropout=0., batch_first = True) for l in range(nlayers)]
        elif sgtype == "LSTM":
            self.rnns = [torch.nn.LSTM(nhid, nhid, dropout=0., batch_first = True) for l in range(nlayers)]
        elif sgtype == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(nhid, nhid, save_prev_x=True, zoneout=0.1, window=2 if l == 0 else 1, output_gate=True, use_cuda=True) for l in range(nlayers)]

        self.rnns = torch.nn.ModuleList(self.rnns)
        print(self.rnns)
        self.sgtype = sgtype
        self.decoder = nn.Linear(nhid, input_features)
        self.nhid = nhid
        self.nlayers = nlayers

        self.dropout = nn.Dropout(p=sg_dropout)

        self.init_weights()
        self.reset_states()

    def init_weights(self):
        self.encoder.bias.data.fill_(0)
        self.decoder.bias.data.fill_(0)
        self.encoder.weight.data.uniform_(-0.1, 0.1)
        self.decoder.weight.data.uniform_(-0.1, 0.1)
        if self.sgtype != 'QRNN':
            for rnn in self.rnns:
                torch.nn.init.orthogonal_(rnn.weight_ih_l0.data)
                torch.nn.init.orthogonal_(rnn.weight_hh_l0.data)

    def forward(self, trigger, context=None):
        data = trigger.reshape(-1, 1, 1)
        data = self.encoder(data * torch.ones_like(data).normal_(mean=1, std=0.5))

        raw_output = data
        new_hidden = []

        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, self.states[l])
            if l != self.nlayers-1:
                raw_output = self.dropout(raw_output)
            new_hidden.append(new_h)
        self.states = self.repackage_hidden(new_hidden)

        raw_output = self.decoder(raw_output)

        result = raw_output.reshape(trigger.shape)

        return result

    def reset_states(self):
        self.states = [None] * self.nlayers
        if self.sgtype == 'QRNN': [r.reset() for r in self.rnns]

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors,
        to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)