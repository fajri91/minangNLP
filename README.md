# MinangNLP

This repository contains three data:
1. Bilingual dictionary: 11,905-size Minangkabau–Indonesian word pairs. (in `resources/kamus_minang_id.dic`)
2. Sentiment Analysis: 5,000-size (1,481 positive and 3,519 negative labels) parallel Minangkabau-Indonesian texts. (in `sentiment/data/folds/`)
3. Machine translation: 16,371-size parallel Minangkabau-Indonesian sentence pairs. (in `translation/wiki_data/`)

Please cite our works if you use our corpus:

 **Fajri Koto, and Ikhwan Koto. _Towards Computational Linguistics in Minangkabau Language: Studies on Sentiment Analysis and Machine Translation_.  In Proceedings of the 34th Pacific Asia Conference on Language, Information and Computation (PACLIC), Vietnam, October, 2020.**
 
 
### Sentiment Analysis Experiment

You need to install dependencies by:
`pip install -r requirements.txt`

For running the experiment, you can run directly:
```
python bilstm.py
python mBert.py
```
We provide jupyter notebook for other models including logistic regression, SVM, and Naive Bayes.

### Machine Translation Experiment

As mentioned in the paper, we use Open-NMT implementation. Please clone and install [this repository](https://github.com/fajri91/OpenNMT-py).

The raw data is in `translation/wiki_data/all_data.xslx`. The provided splits have been tokenized by [Moses Tokenizer](https://pypi.org/project/mosestokenizer/).

For data preprocessing:
```
cd translation/wiki_data
onmt_preprocess -train_src src_train.txt -train_tgt tgt_train.txt -valid_src src_dev.txt -valid_tgt tgt_dev.txt -save_data demo2/demo --share_vocab --src_seq_length 75 --tgt_seq_length 75
```
For training with Bi-LSTM (1 GPU):
```
CUDA_VISIBLE_DEVICES=0 onmt_train -data demo2/demo -save_model demo2/demo-model -world_size 1 
        -gpu_ranks 0 -save_checkpoint_steps 5000 -valid_steps 5000 -train_steps 50000 \
        --encoder_type brnn --decoder_type rnn --enc_rnn_size 200 --dec_rnn_size 200 \
        --copy_attn_force  --reuse_copy_attn  --optim adam --learning_rate 0.001 \
        --warmup_steps 3000 --share_embeddings
```
For training with Transformer (2 GPUs):
```
CUDA_VISIBLE_DEVICES=0,1 python  train.py -data /home/ffajri/Workspace/MinangNLP/translation/data/demo2/demo -save_model /home/ffajri/Workspace/MinangNLP/translation/data/demo2/transformer \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 50000  -max_generator_batches 2 -dropout 0.3 \
        -batch_size 5000 -batch_type tokens -normalization tokens  -accum_count 1 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 5000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
        -world_size 2 -gpu_ranks 0 1 --share_embeddings --master_port 5000
```
For testing (examples):
```
#cd to demo2
CUDA_VISIBLE_DEVICES=0 onmt_translate -gpu 0 -model demo-model_step_10000.pt -src tgt_test.txt -tgt src_test.txt -replace_unk -verbose -share_vocab -output rnn_pred/pred_test10k.txt
```

To evaluate the MT experiment, please use [this](https://github.com/mjpost/sacrebleu).
