import codecs
import os
import re

############## PROPS ##############

a_beam_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# Model
checkpoint_path = '\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\202207111700_log_tests_warm_up_e3_2_6\\checkpoing_3_5000_0.0003\\checkpoint_best_score.pt'
# Data
data_path = '\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\dataset\\processed\\stmc_ph14\\ph14_stmc'
inference_goal_val = "\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\dataset\\raw\\stmc_ph14\\ph14_stmc\\test.de"
# Log
log_dir = '\\Users\\migu1\\git_SLT\\forks\\TIN-SLT\\beam_search_test'

# It's used to transform generated result (in windows may be ansi) in utf, you may not need this
transform_result_to_utf8 = True
###################################

cmd_inference = ("python postprocessing/generate.py {data_path} --remove-bpe --criterion cross_entropy --gen-subset valid --path {checkpoint_path} --beam {beam_size} > {generated_data_file}")
cmd_scoring_bleu_4 = ("python utils/bleu.py 4 {pred} {goal} > {log_file}")
cmd_scoring_rouge = ("python utils/rouge.py {pred} {goal} > {log_file}")

def inference(checkpoint_path, generated_data_file, generated_data_file_utf8, generated_text_file, beam_size):
    cmd_inference_formated = cmd_inference.format(data_path=data_path, checkpoint_path=checkpoint_path, generated_data_file=generated_data_file, beam_size=beam_size)
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

# Make dirs if not exists
if not os.path.exists(log_dir):  
  os.makedirs(log_dir)

for beam_size in a_beam_size:
  generated_data_file = log_dir + '\\generated_data_{beam_size}.out'.format(beam_size=beam_size)
  generated_data_file_utf8 = log_dir + '\\generated_data_{beam_size}_utf8.out'.format(beam_size=beam_size)
  generated_text_file = log_dir + '\\generated_text_valid_{beam_size}.txt'.format(beam_size=beam_size)
  # Inference
  inference(checkpoint_path, generated_data_file, generated_data_file_utf8, generated_text_file, beam_size)
  # Scoring 
  scoring_file_bleu4_txt = log_dir + "\\score_val_blue_{beam_size}.txt".format(beam_size=beam_size)
  scoring_file_rouge_txt = log_dir + "\\score_val_rouge_{beam_size}.txt".format(beam_size=beam_size)
  scoring(generated_text_file, scoring_file_bleu4_txt, scoring_file_rouge_txt)