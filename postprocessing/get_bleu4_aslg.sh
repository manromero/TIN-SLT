data_dir=./../dataset/processed/aslg_add_text_to_gloss_0.38/
ckpt_path=./../checkpoint_save/aslg_84.29/checkpoint74.pt
store_file=./../log/tian_aslg/result_test.out
clean_text_file=./../log/tian_aslg/result_test.txt
BEAM_SIZE=5
BATCH_SIZE=128
python generate.py $data_dir  --instruction_layer_num 1  --batch-size 128 --remove-bpe '' --criterion 'label_smoothed_cross_entropy'  --path $ckpt_path --beam 5 | tee $store_file
grep ^H $store_file | sort -n -k 2 -t '-' | cut -f 3 > $clean_text_file

score_text=./../log/tian_aslg/score_test.txt
goal_test_file=./../dataset/raw/aslg_clean/test.en
python3 ./../utils/bleu.py 1 $clean_text_file $goal_test_file >> $score_text
python3 ./../utils/bleu.py 2 $clean_text_file $goal_test_file >> $score_text
python3 ./../utils/bleu.py 3 $clean_text_file $goal_test_file >> $score_text
python3 ./../utils/bleu.py 4 $clean_text_file $goal_test_file >> $score_text
python3 ./../utils/rouge.py $clean_text_file $goal_test_file | grep 'rouge-l' | grep -Po '\d+\.\d+$' >> $score_text
python3 ./../utils/meteor.py $clean_text_file $goal_test_file >> $score_text
