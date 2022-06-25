import os
import re

############## PROPS ##############

data_path = '\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\dataset\\processed\\stmc_ph14\\ph14_stmc'
save_dir = '\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\checkpoints\\tmp_gloss_de_tmp'
log_file = save_dir + '\\train_log_file.txt'

architecture = 'transformers2'
source = 'gloss'
target = 'de'

optimizer = 'adam'
lr = 0.0003
label_smoothing = 0.3
dropout = 0.45
max_tokens = 4000
min_lr = 1e-09
lr_scheduler = 'inverse_sqrt'
weight_decay = 0.0001
bert_ratio = 0.65
alpha_mode = '\"learn\"'
alpha_bert_strategy = '\"cosine\"'
alpha_bert_max = 1.0
alpha_bert = 1.0
max_epoch = 100
encoder_layers = 4
decoder_layers = 4
bert_model_name = 'dbmdz/bert-base-german-uncased'
criterion = 'label_smoothed_cross_entropy'
max_update = 150000
warmup_updates = 4000
warmup_init_lr = 1e-07
adam_betas = '(0.9,0.98)'

###################################

if not os.path.exists(save_dir):  
  os.makedirs(save_dir)

cmd = ("python train.py {data_path} "
"-a {architecture} --optimizer {optimizer} --lr {lr} -s {source} -t {target} --label-smoothing {label_smoothing} "
"--dropout {dropout} --max-tokens {max_tokens} --min-lr {min_lr} --lr-scheduler {lr_scheduler} --weight-decay {weight_decay} --bert-ratio {bert_ratio} "
"--alpha_mode {alpha_mode} --alpha_bert_strategy {alpha_bert_strategy} --alpha_bert_max {alpha_bert_max} --alpha_bert {alpha_bert} --max-epoch {max_epoch} "
"--encoder-layers {encoder_layers} --decoder-layers {decoder_layers} --bert-model-name {bert_model_name} "
"--criterion {criterion} --max-update {max_update} --warmup-updates {warmup_updates} --warmup-init-lr {warmup_init_lr} "
"--adam-betas {adam_betas} --save-dir {save_dir} --share-all-embeddings > {log_file}").format(
  data_path=data_path, 
  save_dir=save_dir, 
  log_file=log_file,
  architecture=architecture, 
  source=source, 
  target=target, 
  optimizer=optimizer, 
  lr=lr, 
  label_smoothing=label_smoothing, 
  dropout=dropout, 
  max_tokens=max_tokens, 
  min_lr=min_lr, 
  lr_scheduler=lr_scheduler, 
  weight_decay=weight_decay, 
  bert_ratio=bert_ratio, 
  alpha_mode=alpha_mode, 
  alpha_bert_strategy=alpha_bert_strategy, 
  alpha_bert_max=alpha_bert_max, 
  alpha_bert=alpha_bert, 
  max_epoch=max_epoch, 
  encoder_layers=encoder_layers, 
  decoder_layers=decoder_layers, 
  bert_model_name=bert_model_name, 
  criterion=criterion, 
  max_update=max_update, 
  warmup_updates=warmup_updates, 
  warmup_init_lr=warmup_init_lr, 
  adam_betas=adam_betas
)

os.system(cmd)
