# nohup bash ./finetune.sh > logs/finetune.log 2>&1 &

export PYTHONPATH="."

python3 MidiBERT/finetune_bps.py \
    --cuda_devices 0 \
    --batch_size 8 \
    --seed 2021 \
    --dict_file data_creation/prepare_data/dict/CP.pkl \
    --ckpt 'MidiBERT_pretrained.ckpt' \
    --output_dir './BPS_MNID_MI/' \
    --csv_notes_dir '../datasets/Beethoven_motif-main/csv_notes_clean' \
    --mozart_csv_prefix  '../datasets/mozart_pseudo_melody_1105' \
    --mozart_dbeat_dir '../datasets/mozart_sonata_clean_correct_dbeat' \
    --bps_fh_dir  '../datasets/BPS_FH_Dataset' \
    --use_pseudo_label 

