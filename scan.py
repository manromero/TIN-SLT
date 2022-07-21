import codecs
import os
import re

############## PROPS ##############

# Hiperparameters
a_layers = [3]
a_warmup_steps = [5000]
a_learning_rate = [0.0001, 0.0002, 0.0003, 0.0004 ,0.0005, 0.001, 0.01]

# Data
data_path = '\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\dataset\\processed\\stmc_ph14\\ph14_stmc'
inference_goal_val = "\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\dataset\\raw\\stmc_ph14\\ph14_stmc\\valid.de"

# Logs
log_dir = '\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\202207111700_log_tests_learning_rate_e3_4'

# It's used to transform generated result (in windows may be ansi) in utf, you may not need this
transform_result_to_utf8 = True
###################################

architecture = 'transformers2'
source = 'gloss'
target = 'de'

seed=43

# Default hiperparameters
optimizer = 'adam'
# lr = 0.0003
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
# encoder_layers = 4
# decoder_layers = 4
bert_model_name = 'dbmdz/bert-base-german-uncased'
criterion = 'label_smoothed_cross_entropy'
max_update = 150000
# warmup_updates = 4000
warmup_init_lr = 1e-07
adam_betas = '(0.9,0.98)'

# Formats of the commands to train, inference and scoring
cmd_train = ("python trainer/train.py {data_path} "
  "-a {architecture} --optimizer {optimizer} --lr {learning_rate} -s {source} -t {target} --label-smoothing {label_smoothing} "
  "--dropout {dropout} --max-tokens {max_tokens} --min-lr {min_lr} --lr-scheduler {lr_scheduler} --weight-decay {weight_decay} --bert-ratio {bert_ratio} "
  "--alpha_mode {alpha_mode} --alpha_bert_strategy {alpha_bert_strategy} --alpha_bert_max {alpha_bert_max} --alpha_bert {alpha_bert} --max-epoch {max_epoch} "
  "--encoder-layers {layers} --decoder-layers {layers} --bert-model-name {bert_model_name} "
  "--criterion {criterion} --max-update {max_update} --warmup-updates {warmup_steps} --warmup-init-lr {warmup_init_lr} "
  "--adam-betas {adam_betas} --save-dir {checkpoint_dir} --share-all-embeddings --seed {seed} > {log_file}")

cmd_inference = ("python postprocessing/generate.py {data_path} --remove-bpe --criterion cross_entropy --gen-subset valid --path {checkpoint_path} --beam 9 > {generated_data_file}")

cmd_scoring_bleu_4 = ("python utils/bleu.py 4 {pred} {goal} > {log_file}")
cmd_scoring_rouge = ("python utils/rouge.py {pred} {goal} > {log_file}")

def train(layers, warmup_steps, learning_rate, checkpoint_dir, train_log_file):
  cmd_train_formated = cmd_train.format(
    data_path=data_path, 
    log_dir=log_dir, 
    checkpoint_dir=checkpoint_dir,
    log_file=train_log_file,
    architecture=architecture, 
    source=source, 
    target=target, 
    optimizer=optimizer, 
    learning_rate=learning_rate, 
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
    layers=layers, 
    bert_model_name=bert_model_name, 
    criterion=criterion, 
    max_update=max_update, 
    warmup_steps=warmup_steps, 
    warmup_init_lr=warmup_init_lr, 
    adam_betas=adam_betas,
    seed=seed
  )
  os.system(cmd_train_formated)

def inference(checkpoint_path, generated_data_file, generated_data_file_utf8, generated_text_file):
    cmd_inference_formated = cmd_inference.format(data_path=data_path, checkpoint_path=checkpoint_path, generated_data_file=generated_data_file)
    os.system(cmd_inference_formated)
    # INIT - Transform text from ansi to utf-8. Set transform_result_to_utf8 to False if it's not needed
    if transform_result_to_utf8:
      blockSize = 1048576
      with codecs.open(generated_data_file,"r",encoding="mbcs") as sourceFile:
        with codecs.open(generated_data_file_utf8,"w",encoding="UTF-8") as targetFile:
          while True:
            contents = sourceFile.read(blockSize)
            if not contents:
              break
            targetFile.write(contents)
      generated_data_file = generated_data_file_utf8
    # END - Transform text from ansi to utf-8
    # Once the data has been generated, it needs to be cleaned
    # Generate clean text, in sh file = grep ^H result_test.out | sort -n -k 2 -t '-' | cut -f 3 > result_test.txt
    with open(generated_data_file, 'r') as f:
      lines = f.readlines()
      # Filter with lines starts with T-
      translated_text = list(filter(lambda line: line.startswith("H-"), lines))
      # Sort lines with H-NUMBER
      translated_text.sort(key=lambda line: int(line.split("-")[1].split("\t")[0]))
      # obtain result line
      result_lines = list(map(lambda line: line.split("\t")[2], translated_text))

    with open(generated_text_file, 'w') as f:
      for line in result_lines:
        f.write(line)

def scoring(generated_text_file, scoring_file_bleu4_txt, scoring_file_rouge_txt):
  cmd_scoring_bleu_4_formated = cmd_scoring_bleu_4.format(pred=generated_text_file, goal=inference_goal_val, log_file=scoring_file_bleu4_txt)
  os.system(cmd_scoring_bleu_4_formated)
  cmd_scoring_rouge_formated = cmd_scoring_rouge.format(pred=generated_text_file, goal=inference_goal_val, log_file=scoring_file_rouge_txt)
  os.system(cmd_scoring_rouge_formated)

############################################################ MAIN ############################################################

# Make dirs if not exists
if not os.path.exists(log_dir):  
  os.makedirs(log_dir)

for layers in a_layers:
    for warmup_steps in a_warmup_steps:
      for learning_rate in a_learning_rate:   
        checkpoint_dir = log_dir + "\\checkpoing_{layers}_{warmup_steps}_{learning_rate}".format(layers=layers, warmup_steps=warmup_steps, learning_rate=learning_rate)
        # Make dirs if not exists
        if not os.path.exists(checkpoint_dir):  
          os.makedirs(checkpoint_dir)
        # Train
        train_log_file = log_dir + "\\train_log_file_{layers}_{warmup_steps}_{learning_rate}.txt".format(layers=layers, warmup_steps=warmup_steps, learning_rate=learning_rate)
        train(layers, warmup_steps, learning_rate, checkpoint_dir, train_log_file)
        # Inference
        checkpoint_path = checkpoint_dir + "\\checkpoint_best_score.pt"
        generated_data_file = log_dir + "\\generated_data_{layers}_{warmup_steps}_{learning_rate}.out".format(layers=layers, warmup_steps=warmup_steps, learning_rate=learning_rate)
        generated_data_file_utf8 = log_dir + "\\generated_data_{layers}_{warmup_steps}_{learning_rate}_utf8.out".format(layers=layers, warmup_steps=warmup_steps, learning_rate=learning_rate)
        generated_text_file = log_dir + "\\generated_text_valid_{layers}_{warmup_steps}_{learning_rate}.txt".format(layers=layers, warmup_steps=warmup_steps, learning_rate=learning_rate)
        inference(checkpoint_path, generated_data_file, generated_data_file_utf8, generated_text_file)
        # Scoring
        scoring_file_bleu4_txt = log_dir + "\\score_val_blue_{layers}_{warmup_steps}_{learning_rate}.txt".format(layers=layers, warmup_steps=warmup_steps, learning_rate=learning_rate)
        scoring_file_rouge_txt = log_dir + "\\score_val_rouge_{layers}_{warmup_steps}_{learning_rate}.txt".format(layers=layers, warmup_steps=warmup_steps, learning_rate=learning_rate)
        scoring(generated_text_file, scoring_file_bleu4_txt, scoring_file_rouge_txt)