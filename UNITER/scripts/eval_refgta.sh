OUT_DIR=$1 
ROOT_DIR=/PATH/TO/DIRECTORY/CONTAINING/FEATURE/FILES/
python inf_re.py \
    --txt_db /PATH/TO/refgta_val.jsonl \
    --img_db $ROOT_DIR/refgta_val_gt_boxes10100.pt \
    --output_dir $OUT_DIR \
    --checkpoint best \
    --tmp_file re_exp/tmp_refcocog.txt \
    --simple_format --n_workers 1 --batch_size 128

python inf_re.py \
    --txt_db /PATH/TO/refgta_val.jsonl \
    --img_db $ROOT_DIR/refgta_val_unidet_dt_boxes10100.pt \
    --output_dir $OUT_DIR \
    --checkpoint best \
    --tmp_file re_exp/tmp_refcocog.txt \
    --simple_format --n_workers 1 --batch_size 128

python inf_re.py \
    --txt_db /PATH/TO/refgta_val.jsonl \
    --img_db $ROOT_DIR/refgta_val_unidet_all_dt_boxes10100.pt \
    --output_dir $OUT_DIR \
    --checkpoint best \
    --tmp_file re_exp/tmp_refcocog.txt \
    --simple_format --n_workers 1 --batch_size 128
