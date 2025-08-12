# TISMIR hybrid motif discovery

This is the official implementation of the following paper:

Jun-You Wang, Yu-Chia Kuo, and Li Su, "Improving Motif Discovery of Symbolic Polyphonic Music with Motif Note Identification," *Transactions of the International Society for Music Information Retrieval*, 2025.

## MNID (motif note identification)

*Modified from [wazenmai/MIDI-BERT](https://github.com/wazenmai/MIDI-BERT).

First, download the pretrained MidiBERT model from [MIDI-BERT's repo](https://github.com/wazenmai/MIDI-BERT), and put it somewhere (e.g., `MNID/MidiBERT_pretrained.ckpt`).

### Training w/ the proposed MI method

Modify `--ckpt` in `finetune.sh`  to the pretrained MidiBERT model's path (or you can also simply put the pretrained checkpoint to `MNID/MidiBERT_pretrained.ckpt`).

```
cd MNID
bash finetune.sh
```

The output models will be written to `--output_dir`, so modify it in `finetune.sh` to customize it.

###### Regarding MPS pseudo-labels

We have already extracted the pseudo-labels from the MPS dataset and put them to `../datasets/mozart_pseudo_melody_1105_{0,1,2,3,4}` (for the **MI** setting) and `../datasets/mozart_pseudo_1105_{0,1,2,3,4}` (for the **PL** setting). 

The reason we have five versions of pseudo-labels is that generating pseudo-labels involves the use of a pretrained MNID model (the first MNID model). The train/valid/test setting of the first and the second MNID models should remain the same, e.g., for the first fold's **MI** setting, the pseudo-labeler is also trained with the first fold's setting, i.e., excluding song [1, 2, 3, 4, 5, 6, 7] as the test set. Therefore, we need to generate five different pseudo-labels for MPS for five folds. **In short, we do this to ensure that the model cannot "see" the test split at any time of the training pipeline.**

Actually, the pseudo-labeling step is difficult to reproduce because the source code of (Simonetta et al., 2019) used `theano` and `python 2.7` to train the NN model on CPU (!!!), see [LIMUNIMI/Symbolic-Melody-Identification](https://github.com/LIMUNIMI/Symbolic-Melody-Identification). It is not compatible to our MNID code environment so another environment must be built for it. If you are interested in it, just visit their Github repo. For the sake of reproducibility, we still release the MeloID models and our 5-fold cross-validation dataset partitioning in `mozart_melody/`.

### Training w/o pseudo-label (no MPS dataset)

Remove`--use_pseudo_label` flag in `finetune.sh`. Then,

```
cd MNID
bash finetune.sh
```

### Evaluation (BPS-motif)

```
python extract_csv_ft.py --input_dir <DIR_TO_CSV_NOTES> \
    --bps_fh_dir <DIR_TO_BPS_FH> --output_csv_dir <OUTPUT_DIR> \
    --ckpt_dir <DIR_TO_MNID_MODEL_CKPT>
```

where `ckpt_dir` should be the same as the `output_dir` in `finetune.sh`. This script reads note lists from `input_dir` (should be set to `../datasets/Beethoven_motif-main/csv_notes_clean` if nothing is changed) and performs MNID. Then, the outputs (the list of predicted motif notes) are written to `output_csv_dir`. 

By the way, the reason of using the BPS-FH dataset (`bps_fh_dir`) is to obtain the downbeat information from the `dBeats.xlsx` files. It should be set to `../datasets/BPS_FH_Dataset` if you didn't modify its path.

This script also outputs evaluation results in the following format:

```
Test 5-fold overall mean F1: 0.7200 Acc: 0.8365 P: 0.7010 R: 0.7634
Std for 5-fold F1: 0.0574 Acc: 0.0300 P: 0.0730 R: 0.0695
```

### Evaluation (JKU-PDD)

The evaluation/inference on JKU-PDD is slightly different from BPS-motif because we use the average prediction of five MNID models (one from each fold) to perform MNID on JKU-PDD.

```
python extract_csv_ft_jku.py --input_dir <DIR_TO_CSV_NOTES> \
    --db_dir <DIR_TO_DBEAT_FILES> --output_csv_dir <OUTPUT_DIR> \
    --ckpt_dir <DIR_TO_MNID_MODEL_CKPT>
```

where `ckpt_dir` should be the same as the `output_dir` in `finetune.sh`. If nothing is modified, then `input_dir` should be `../datasets/jkupdd_clean`, and `db_dir` should be`../datasets/jkupdd_clean_dbeats`. These are the files that we have proprocessed on JKU-PDD.

## Motif discovery (BPS-motif)

```
cd motif_discovery
python experiments.py --csv_note_dir <DIR_TO_CSV_NOTES> \
    --method <METHOD> \
    [--csv_label_dir <DIR_TO_LABELS>] \
    [--motif_midi_dir <DIR_TO_MIDI>] \
```

Arguments:

- `--csv_note_dir` : Path to the directory containing csv note files (output of the MNID model).
- `--method`: Choose which algorithm to run (CSA, SIATEC).
- `--csv_label_dir` (optional): Path to the directory of motif label csv files. 
- `--motif_midi_dir` (optional): Path to the directory of motif MIDI files.

If nothing is modified, then `csv_label_dir` should be `../datasets/Beethoven_motif-main/csv_label`, while `motif_midi_dir` should be `../datasets/Beethoven_motif-main/motif_midi`. To run the non-MNID baseline, simply set `csv_note_dir` to `../datasets/Beethoven_motif-main/csv_notes_clean`.

**IMPORTANT**: It is normal to take more than one day to perform motif discovery.

## Motif discovery (JKU-PDD)

```
cd motif_discovery
python experiments_jku.py --csv_note_dir <DIR_TO_CSV_NOTES> \
    --method <METHOD>
```

Arguments:

- `--csv_note_dir` : Path to the directory containing csv note files (output of the MNID model).
- `--method`: Choose which algorithm to run (CSA, SIATEC).

**IMPORTANT**: This code should be finished pretty soon because the scale of JKU-PDD is not that large (compared to BPS-motif).
