from models.TENER import TENER
from fastNLP.embeddings import CNNCharEmbedding
from fastNLP import cache_results
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from fastNLP import SpanFPreRecMetric, BucketSampler
from fastNLP.io.pipe.conll import OntoNotesNERPipe
from fastNLP.embeddings import StaticEmbedding, StackEmbedding, LSTMCharEmbedding
from modules.TransformerEmbedding import TransformerCharEmbed
from modules.pipe import Conll2003NERPipe

import argparse
from modules.callbacks import EvaluateCallback

device = 0
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='en-ontonotes', choices=['conll2003', 'en-ontonotes'])

args = parser.parse_args()

dataset = args.dataset

if dataset == 'conll2003':
    n_heads = 14
    head_dims = 128
    num_layers = 2
    lr = 0.0009
    attn_type = 'adatrans'
    char_type = 'cnn'
elif dataset == 'en-ontonotes':
    n_heads =  8
    head_dims = 96
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    char_type = 'adatrans'

pos_embed = None

#########hyper
batch_size = 16
warmup_steps = 0.01
after_norm = 1
model_type = 'transformer'
normalize_embed = True
#########hyper

dropout=0.15
fc_dropout=0.4

encoding_type = 'bioes'
name = 'caches/{}_{}_{}_{}_{}.pkl'.format(dataset, model_type, encoding_type, char_type, normalize_embed)
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)



@cache_results(name, _refresh=False)
def load_data():
    # 替换路径
    if dataset == 'conll2003':
        # conll2003的lr不能超过0.002
        paths = {'test': "../data/conll2003/test.txt",
                 'train': "../data/conll2003/train.txt",
                 'dev': "../data/conll2003/dev.txt"}
        data = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(paths)
    elif dataset == 'en-ontonotes':
        # 会使用这个文件夹下的train.txt, test.txt, dev.txt等文件
        paths = '../data/en-ontonotes/english'
        data = OntoNotesNERPipe(encoding_type=encoding_type).process_from_file(paths)
    char_embed = None
    if char_type == 'cnn':
        char_embed = CNNCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, filter_nums=[30],
                                      kernel_sizes=[3], word_dropout=0, dropout=0.3, pool_method='max'
                                      , include_word_start_end=False, min_char_freq=2)
    elif char_type in ['adatrans', 'naive']:
        char_embed = TransformerCharEmbed(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, word_dropout=0,
                 dropout=0.3, pool_method='max', activation='relu',
                 min_char_freq=2, requires_grad=True, include_word_start_end=False,
                 char_attn_type=char_type, char_n_head=3, char_dim_ffn=60, char_scale=char_type=='naive',
                 char_dropout=0.15, char_after_norm=True)
    elif char_type == 'lstm':
        char_embed = LSTMCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, word_dropout=0,
                 dropout=0.3, hidden_size=100, pool_method='max', activation='relu',
                 min_char_freq=2, bidirectional=True, requires_grad=True, include_word_start_end=False)
    word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
                                 model_dir_or_name='en-glove-6b-100d',
                                 requires_grad=True, lower=True, word_dropout=0, dropout=0.5,
                                 only_norm_found_vector=normalize_embed)
    if char_embed is not None:
        embed = StackEmbedding([word_embed, char_embed], dropout=0, word_dropout=0.02)
    else:
        word_embed.word_drop = 0.02
        embed = word_embed

    data.rename_field('words', 'chars')
    return data, embed

data_bundle, embed = load_data()
print(data_bundle)

model = TENER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
                       d_model=d_model, n_head=n_heads,
                       feedforward_dim=dim_feedforward, dropout=dropout,
                        after_norm=after_norm, attn_type=attn_type,
                       bi_embed=None,
                        fc_dropout=fc_dropout,
                       pos_embed=pos_embed,
              scale=attn_type=='transformer')

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(data_bundle.get_dataset('test'))

if warmup_steps>0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, evaluate_callback])

trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer, batch_size=batch_size, sampler=BucketSampler(),
                  num_workers=2, n_epochs=100, dev_data=data_bundle.get_dataset('dev'),
                  metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type),
                  dev_batch_size=batch_size*5, callbacks=callbacks, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=300, save_path=None)
trainer.train(load_best_model=False)
