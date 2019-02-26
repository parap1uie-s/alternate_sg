# alternate_sg

This repository contains the code used for the paper: Alternating Synthetic and Real Gradients for Neural Language Modeling

## Software Requirements

- python 3+

- PyTorch 0.4

- \* [PyTorch-QRNN 0.2.1 ](https://github.com/salesforce/pytorch-qrnn)

\* Bug fixed, replace site-packages/torchqrnn/forget_mult.py with forget_mult.py after qrnn installed.

## Experiments

The element-wise DNI module interact with base module by sending gradients to the hidden states of base module at truncated point.

### Word level Penn Treebank (PTB) with QRNN(base) and QRNN(DNI)

- DNI

`python3.6 -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 42 --dropouti 0.4 --optimizer adam --lr 1e-3 --cuda --bptt=200 --epochs=320 --expname=ptbword_qrnn_sg --sg_hidden_num=50 --sg_nlayers=1 --sg_dropout=0.1 --whenswitch=2 --use_sg --sgtype=QRNN`

- DNI with Restart base == 160

`python3.6 -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 42 --dropouti 0.4 --optimizer adam --lr 1e-3 --cuda --bptt=200 --epochs=320 --expname=ptbword_qrnn_sg_restart_rb160 --sg_hidden_num=50 --sg_nlayers=1 --sg_dropout=0.1 --whenswitch=2 --use_sg --sgtype=QRNN --restart_base 160 320 --restart_lr`

- DNI with Restart base == 5

`python3.6 -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 42 --dropouti 0.4 --optimizer adam --lr 1e-3 --cuda --bptt=200 --epochs=320 --expname=ptbword_qrnn_sg_restart_rb5 --sg_hidden_num=50 --sg_nlayers=1 --sg_dropout=0.1 --whenswitch=2 --use_sg --sgtype=QRNN --restart_base=5 --restart_lr`

- Alternating DNI with Restart base == 160

`python3.6 -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 42 --dropouti 0.4 --optimizer adam --lr 1e-3 --cuda --bptt=200 --epochs=320 --expname=ptbword_qrnn_mixture_rb160 --sg_hidden_num=50 --sg_nlayers=1 --sg_dropout=0.1 --use_sg --sgtype=QRNN --mixture --restart_base 160 320 --restart_lr`

- Alternating DNI with Restart base == 5

`python3.6 -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 42 --dropouti 0.4 --optimizer adam --lr 1e-3 --cuda --bptt=200 --epochs=320 --expname=ptbword_qrnn_mixture_rb5 --sg_hidden_num=50 --sg_nlayers=1 --sg_dropout=0.1 --use_sg --sgtype=QRNN --mixture --restart_base=5 --restart_lr`

### Word level WikiText-2 (WT2) with QRNN(base) and QRNN(DNI)

- DNI

`python3.6 -u main.py --cuda --bptt=200 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.2 --nhid 1550 --nlayers 4 --seed 42 --model QRNN --wdrop 0.1 --batch_size 20 --optimizer adam --lr 1e-3 --expname=wikiword_qrnn_sg --epochs 320 --sg_hidden_num=50 --sg_nlayers=1 --sg_dropout=0.1 --whenswitch=2 --use_sg --sgtype=QRNN`

- DNI with Restart base == 160

`python3.6 -u main.py --cuda --bptt=200 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.2 --nhid 1550 --nlayers 4 --seed 42 --model QRNN --wdrop 0.1 --batch_size 20 --optimizer adam --lr 1e-3 --expname=wikiword_qrnn_sg_restart_rb160 --epochs 320 --sg_hidden_num=50 --sg_nlayers=1 --sg_dropout=0.1 --whenswitch=2 --use_sg --sgtype=QRNN --restart_base 160 320 --restart_lr`

- DNI with Restart base == 5

`python3.6 -u main.py --cuda --bptt=200 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.2 --nhid 1550 --nlayers 4 --seed 42 --model QRNN --wdrop 0.1 --batch_size 20 --optimizer adam --lr 1e-3 --expname=wikiword_qrnn_sg_restart_rb5 --epochs 320 --sg_hidden_num=50 --sg_nlayers=1 --sg_dropout=0.1 --whenswitch=2 --use_sg --sgtype=QRNN --restart_base=5 --restart_lr`

- Alternating DNI with Restart base == 160

`python3.6 -u main.py --cuda --bptt=200 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.2 --nhid 1550 --nlayers 4 --seed 42 --model QRNN --wdrop 0.1 --batch_size 20 --optimizer adam --lr 1e-3 --expname=wikiword_qrnn_mixture_rb160 --epochs 320 --sg_hidden_num=50 --sg_nlayers=1 --sg_dropout=0.1 --use_sg --sgtype=QRNN --mixture --restart_base 160 320 --restart_lr --whenswitch=2`

- Alternating DNI with Restart base == 5

`python3.6 -u main.py --cuda --bptt=200 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.2 --nhid 1550 --nlayers 4 --seed 42 --model QRNN --wdrop 0.1 --batch_size 20 --optimizer adam --lr 1e-3 --expname=wikiword_qrnn_mixture_rb5 --epochs 320 --sg_hidden_num=50 --sg_nlayers=1 --sg_dropout=0.1 --use_sg --sgtype=QRNN --mixture --restart_base=5 --restart_lr --whenswitch=2`

## hyper parameters

### ptbword
| name  | BPTT | DNI |
|:---:|:---:|:---:|
| lr  | 1e-3 | 1e-3 |
| optimizer  | adam | adam |
| BPTT length | 200  | 200 |
| base model type  | QRNN | QRNN |
| base neurons per layer  | 1550 | 1550 |
| base layers  | 4 | 4 |
| random seed  | 42 | 42 |
| batch size  | 20 | 20 |
| DNI model type  | N/A | QRNN |
| DNI neurons / layer(s)  | N/A | 50/1 |

### wikiword-2
| name  | BPTT | DNI |
|:---:|:---:|:---:|
| lr  | 1e-3 | 1e-3 |
| optimizer  | adam | adam |
| BPTT length | 200  | 200 |
| base model type  | QRNN | QRNN |
| base neurons per layer  | 1550 | 1550 |
| base layers  | 4 | 4 |
| random seed  | 42 | 42 |
| batch size  | 20 | 20 |
| DNI model type  | N/A | QRNN |
| DNI neurons / layer(s)  | N/A | 50/1 |
