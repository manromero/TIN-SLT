import codecs
import os
import re

############## PROPS ##############

data_path = '\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\dataset\\processed\\stmc_ph14\\ph14_stmc'
checkpoint_path = '\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\checkpoints\\tmp_gloss_de_tmp\\checkpoint_best_score.pt'

log_dir = '\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\evaluations\\tmp_gloss_de_tmp'

generated_data_file = log_dir + '\\result_test.out'
# It's used to transform generated result (in windows may be ansi) in utf, you may not need this
transform_result_to_utf8 = True
generated_data_file_utf8 = log_dir + '\\result_test_utf8.out'
generated_text_file = log_dir + '\\generate_text.txt'
goal_test_file = '\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\dataset\\raw\\stmc_ph14\\ph14_stmc\\test.de'
score_file = log_dir + '\\score_file.txt'

bert_model_name = 'dbmdz/bert-base-german-uncased'

###################################

if not os.path.exists(log_dir):  
  os.makedirs(log_dir)

cmd_generate_text = ("python generate.py {data_path} "
"--remove-bpe --criterion cross_entropy --path {checkpoint_path} --beam 9 "
"> {generated_data_file}").format(
    data_path=data_path,
    checkpoint_path=checkpoint_path,
    generated_data_file=generated_data_file
)

os.system(cmd_generate_text)

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

# Generate bleu 4 file
cmd_generate_bleu_4 = ("python ./../utils/bleu.py 4 {generated_text_file} {goal_test_file} > {score_file}").format(
  generated_text_file=generated_text_file, 
  goal_test_file=goal_test_file, 
  score_file=score_file
)

os.system(cmd_generate_bleu_4)