OUT_DIR=$1 
ROOT_DIR=/PATH/TO/DIRECTORY/CONTAINING/FEATURE/FILES

python inf_re.py \
    --txt_db /PATH/TO/refcoco+_val.jsonl \
    --img_db $ROOT_DIR/refcoco+_val_gt_boxes10100.pt \
    --output_dir $OUT_DIR \
    --checkpoint best \
    --simple_format --n_workers 1 --batch_size 128 --tmp_file re_exp/tmp_refcoco+.txt

python inf_re.py \
	--txt_db /PATH/TO/refcoco+_val.jsonl \
	--img_db $ROOT_DIR/refcoco+_val_dt_boxes10100.pt \
        --output_dir $OUT_DIR  \
        --checkpoint best \
	--simple_format --n_workers 1 --batch_size 128 --tmp_file re_exp/tmp_refcoco+.txt
