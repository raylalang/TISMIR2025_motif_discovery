import argparse
import numpy as np
import pickle
import os, csv, sys
from pathlib import Path

import random
import openpyxl
from melody_extraction.midibert.utils import *

from torch.utils.data import DataLoader
import torch
from transformers import BertConfig

from MidiBERT.model import MidiBert
from MidiBERT.finetune_trainer import FinetuneTrainer
from MidiBERT.finetune_dataset import FinetuneDataset

from matplotlib import pyplot as plt
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="")

    ### path setup ###
    parser.add_argument(
        "--dict_file", type=str, default="data_creation/prepare_data/dict/CP.pkl"
    )
    parser.add_argument("--ckpt", default="pretrained.ckpt")
    parser.add_argument("--output_dir", type=str)
    # Add these path arguments:
    parser.add_argument(
        "--csv_notes_dir",
        type=str,
        default="../datasets/Beethoven_motif-main/csv_notes_clean",
    )
    parser.add_argument("--bps_fh_dir", type=str, default="../datasets/BPS_FH_Dataset")
    parser.add_argument(
        "--mozart_csv_prefix", type=str, default="../datasets/mozart_pseudo_melody_1105"
    )
    parser.add_argument(
        "--mozart_dbeat_dir",
        type=str,
        default="../datasets/mozart_sonata_clean_correct_dbeat",
    )

    parser.add_argument("--use_pseudo_label", action="store_true")

    ### parameter setting ###
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="all sequences are padded to `max_seq_len`",
    )
    parser.add_argument("--hs", type=int, default=768)
    parser.add_argument("--index_layer", type=int, default=12, help="number of layers")
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="initial learning rate")
    parser.add_argument("--nopretrain", action="store_true")  # default: false

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default=False
    parser.add_argument(
        "--cuda_devices",
        type=int,
        nargs="+",
        default=[
            0,
        ],
        help="CUDA device ids",
    )

    args = parser.parse_args()
    args.task = "melody"
    args.class_num = 4
    return args


def extract_events_from_csv(csv_note_path, downbeat_csv_path, negative_pitch_shift=0):
    # Get notes
    raw_information = []
    notes = []
    with open(csv_note_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if row[0] != "onset":
                onset = float(int(float(row[0]) * 100.0)) / 100
                duration = float(int(float(row[3]) * 100.0)) / 100

                if row[-1] == "":
                    note = [
                        onset,
                        max(min(int(row[1]) + int(negative_pitch_shift), 107), 24),
                        duration,
                    ]
                else:
                    note = [onset, int(row[1]), duration]
                if duration > 0:
                    notes.append(note)
                    raw_information.append(row)

    dbeats = [0.0]
    suffix = Path(downbeat_csv_path).suffix.lower()

    if suffix == ".xlsx":
        wb = openpyxl.load_workbook(downbeat_csv_path, read_only=True, data_only=True)
        ws = wb.worksheets[0]  # first sheet

        # read first column
        for (v,) in ws.iter_rows(min_col=1, max_col=1, values_only=True):
            if v is None:
                continue
            try:
                dbeats.append(float(v))
            except (TypeError, ValueError):
                # e.g., header row like "downbeat"
                continue

    elif suffix == ".csv":
        with open(downbeat_csv_path, "r", newline="") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                if not row:
                    continue
                try:
                    dbeats.append(float(row[0]))
                except (TypeError, ValueError):
                    # header / bad row
                    continue
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    dbeats[0] = dbeats[1] - (dbeats[2] - dbeats[1])

    dbeats.append(dbeats[-1] + (dbeats[-1] - dbeats[-2]))

    groups = []
    cur_downbeat_timing = 0
    # print (dbeats)
    note_count = 0
    for db1, db2 in zip(dbeats[:-1], dbeats[1:]):
        insiders = []
        if cur_downbeat_timing == 0:
            insiders.append(
                Item(name="Tempo", start=0, end=None, velocity=None, pitch=120)
            )

        for i in range(len(notes)):
            if (notes[i][0] >= db1) and (notes[i][0] < db2):
                start_interp_in_bar = max((notes[i][0] - db1) / (db2 - db1), 0)
                end_interp_in_bar = (notes[i][0] + notes[i][2] - db1) / (db2 - db1)

                insiders.append(
                    Item(
                        name="Note",
                        start=cur_downbeat_timing
                        + start_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        end=cur_downbeat_timing
                        + end_interp_in_bar * (DEFAULT_RESOLUTION * 4),
                        velocity=64,
                        pitch=notes[i][1],
                    )
                )
                note_count = note_count + 1
        overall = (
            [cur_downbeat_timing]
            + insiders
            + [cur_downbeat_timing + DEFAULT_RESOLUTION * 4]
        )
        cur_downbeat_timing = cur_downbeat_timing + DEFAULT_RESOLUTION * 4
        groups.append(overall)

    events = item2event(groups)
    return events, raw_information


def extract_events(input_path):
    note_items, tempo_items = read_items(input_path)
    # print (note_items)
    if len(note_items) == 0:  # if the midi contains nothing
        return None
    note_items = quantize_items(note_items)
    max_time = note_items[-1].end
    items = tempo_items + note_items

    groups = group_items(items, max_time)
    events = item2event(groups)
    return events


def padding(data, pad_word, max_len):
    pad_len = max_len - len(data)
    for _ in range(pad_len):
        data.append(pad_word)

    return data


def prepare_data_from_csv(
    csv_note_path,
    downbeat_csv_path,
    pad_word,
    e2w,
    negative_pitch_shift,
    offset,
    max_len=512,
):
    """
    Prepare data for a single midi
    """
    # extract events
    events, raw_information = extract_events_from_csv(
        csv_note_path, downbeat_csv_path, negative_pitch_shift=negative_pitch_shift
    )
    # print (len(events), len(raw_information))
    if not events:  # if midi contains nothing
        raise ValueError(f"The given {csv_note_path} is empty")

    # events to words
    # 1. Bar, Position, Pitch, Duration, Velocity ---> we only convert note events to words
    # 2. Position, Tempo Style, Tempo Class
    # 3. Bar
    words = []

    for tup in events:
        nts = []
        if len(tup) == 5:  # Note
            for e in tup:
                if e.name == "Velocity":
                    continue
                e_text = "{} {}".format(e.name, e.value)
                nts.append(e2w[e.name][e_text])
            words.append(nts)

    # slice to chunks so that max length = max_len (default: 512)
    slice_words = []
    # for i in range(0, len(words), max_len):
    #     slice_words.append(words[i:i+max_len])
    for i in range(-offset, len(words), max_len):
        if i < 0:
            data = []
            pad_len = -i
            for _ in range(pad_len):
                data.append(pad_word)
            for j in range(i + max_len):
                data.append(words[j])
            slice_words.append(data)
        else:
            slice_words.append(words[i : i + max_len])

    # padding or drop
    if len(slice_words[-1]) < max_len:
        slice_words[-1] = padding(slice_words[-1], pad_word, max_len)

    slice_words = np.array(slice_words)

    return events, slice_words, raw_information


def read_one_split(indices, pad_word, e2w, args):
    tokens = []
    all_gt_labels = []
    for indice in indices:
        piece = str(indice + 1).zfill(2)
        csv_note_path = os.path.join(args.csv_notes_dir, piece + "-1.csv")
        downbeat_csv_path = os.path.join(
            args.bps_fh_dir, str(indice + 1), "dBeats.xlsx"
        )

        if indice + 1 == 8:
            downbeat_csv_path = downbeat_csv_path.replace("dBeats", "dbeats")

        events, slice_words, raw_information = prepare_data_from_csv(
            csv_note_path,
            downbeat_csv_path,
            pad_word,
            e2w,
            negative_pitch_shift=0,
            offset=0,
            max_len=512,
        )
        slice_words = torch.from_numpy(slice_words)

        note_ind = 0
        note_raw_information_count = 0

        for i in range(len(slice_words)):
            gt_labels = []
            for j in range(len(slice_words[i])):
                if slice_words[i][j][0] == 0 or slice_words[i][j][0] == 1:
                    if raw_information[note_ind][-1] != "":
                        gt_labels.append(1)
                    else:
                        gt_labels.append(3)
                    note_ind += 1
                else:
                    gt_labels.append(0)
            gt_labels = torch.tensor(gt_labels).unsqueeze(0)
            all_gt_labels.append(gt_labels)

        tokens.append(slice_words)

    tokens = torch.cat(tokens, dim=0).numpy()
    all_gt_labels = torch.cat(all_gt_labels, dim=0).numpy()
    return tokens, all_gt_labels


def read_one_split_aug(indices, pad_word, e2w, args):
    tokens = []
    all_gt_labels = []
    offsets = [0, 256]
    for indice in tqdm(indices):
        piece = str(indice + 1).zfill(2)
        csv_note_path = os.path.join(args.csv_notes_dir, piece + "-1.csv")
        downbeat_csv_path = os.path.join(
            args.bps_fh_dir, str(indice + 1), "dBeats.xlsx"
        )

        if indice + 1 == 8:
            downbeat_csv_path = downbeat_csv_path.replace("dBeats", "dbeats")

        if torch.rand(1) >= 0.75:
            pitch_shift_list = [0, 12]
        elif torch.rand(1) >= 0.5:
            pitch_shift_list = [0, -12]
        else:
            pitch_shift_list = [
                0,
            ]

        for negative_pitch_shift in pitch_shift_list:
            for offset in offsets:
                events, slice_words, raw_information = prepare_data_from_csv(
                    csv_note_path,
                    downbeat_csv_path,
                    pad_word,
                    e2w,
                    negative_pitch_shift=negative_pitch_shift,
                    offset=offset,
                    max_len=512,
                )
                slice_words = torch.from_numpy(slice_words)

                note_ind = 0
                note_raw_information_count = 0

                for i in range(len(slice_words)):
                    gt_labels = []
                    for j in range(len(slice_words[i])):
                        if slice_words[i][j][0] == 0 or slice_words[i][j][0] == 1:
                            if raw_information[note_ind][-1] != "":
                                gt_labels.append(1)
                            else:
                                gt_labels.append(3)
                            note_ind += 1
                        else:
                            gt_labels.append(0)
                    gt_labels = torch.tensor(gt_labels).unsqueeze(0)
                    all_gt_labels.append(gt_labels)

                tokens.append(slice_words)

    tokens = torch.cat(tokens, dim=0).numpy()
    all_gt_labels = torch.cat(all_gt_labels, dim=0).numpy()
    return tokens, all_gt_labels


def read_mozart_aug(indices, pad_word, e2w, fold_id, args):
    tokens = []
    all_gt_labels = []
    offsets = [0, 256]
    for indice in tqdm(indices):
        piece = str(indice + 1).zfill(2)

        csv_dir = f"{args.mozart_csv_prefix}_{str(fold_id)}"
        csv_note_path = os.path.join(csv_dir, f"{piece}-1.csv")
        downbeat_csv_path = os.path.join(args.mozart_dbeat_dir, piece + "-1.csv")

        # 25% of the time, we will apply pitch shift augmentation with +12 semitones
        # Another 25% of the time, we will apply pitch shift augmentation with -12 semitones
        if torch.rand(1) >= 0.75:
            pitch_shift_list = [0, 12]
        elif torch.rand(1) >= 0.5:
            pitch_shift_list = [0, -12]
        else:
            pitch_shift_list = [
                0,
            ]

        for negative_pitch_shift in pitch_shift_list:
            for offset in offsets:
                events, slice_words, raw_information = prepare_data_from_csv(
                    csv_note_path,
                    downbeat_csv_path,
                    pad_word,
                    e2w,
                    negative_pitch_shift=negative_pitch_shift,
                    offset=offset,
                    max_len=512,
                )
                slice_words = torch.from_numpy(slice_words)

                note_ind = 0
                note_raw_information_count = 0

                for i in range(len(slice_words)):
                    gt_labels = []
                    for j in range(len(slice_words[i])):
                        if slice_words[i][j][0] == 0 or slice_words[i][j][0] == 1:
                            if raw_information[note_ind][-1] != "":
                                gt_labels.append(1)
                            else:
                                gt_labels.append(3)
                            note_ind += 1
                        else:
                            gt_labels.append(0)
                    gt_labels = torch.tensor(gt_labels).unsqueeze(0)
                    all_gt_labels.append(gt_labels)

                tokens.append(slice_words)

    tokens = torch.cat(tokens, dim=0).numpy()
    all_gt_labels = torch.cat(all_gt_labels, dim=0).numpy()
    print(tokens.shape, all_gt_labels.shape)
    return tokens, all_gt_labels


def load_motif_data(
    mozart_lookup,
    train_indices,
    valid_indices,
    test_indices,
    pad_word,
    e2w,
    fold_id,
    args,
):
    x_train, y_train = read_one_split_aug(train_indices, pad_word, e2w, args)

    if args.use_pseudo_label:
        # If using pseudo labels, we will read the Mozart dataset
        print("Using pseudo labels from Mozart dataset")
        x_mozart_train, y_mozart_train = read_mozart_aug(
            mozart_lookup, pad_word, e2w, fold_id, args
        )
        x_train = np.concatenate((x_train, x_mozart_train), axis=0)
        y_train = np.concatenate((y_train, y_mozart_train), axis=0)

    x_valid, y_valid = read_one_split(valid_indices, pad_word, e2w, args)
    x_test, y_test = read_one_split(test_indices, pad_word, e2w, args)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def train_model(
    args,
    mozart_lookup,
    train_indices,
    valid_indices,
    test_indices,
    pad_word,
    e2w,
    w2e,
    output_dir,
    fold_id,
):
    seq_class = False
    os.makedirs(os.path.join(output_dir, fold_id), exist_ok=True)

    print("\nLoading Dataset")
    X_train, X_val, X_test, y_train, y_val, y_test = load_motif_data(
        mozart_lookup,
        train_indices,
        valid_indices,
        test_indices,
        pad_word,
        e2w,
        fold_id,
        args,
    )

    trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val)
    testset = FinetuneDataset(X=X_test, y=y_test)

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(
        validset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print("   len of valid_loader", len(valid_loader))
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print("   len of test_loader", len(test_loader))

    print("\nBuilding BERT model")
    configuration = BertConfig(
        max_position_embeddings=args.max_seq_len,
        position_embedding_type="relative_key_query",
        hidden_size=args.hs,
    )

    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    best_mdl = ""
    if not args.nopretrain:
        best_mdl = args.ckpt
        print("   Loading pre-trained model from", best_mdl.split("/")[-1])
        checkpoint = torch.load(best_mdl, map_location="cpu")
        midibert.load_state_dict(checkpoint["state_dict"], strict=False)

    index_layer = int(args.index_layer) - 13
    print("\nCreating Finetune Trainer using index layer", index_layer)
    trainer = FinetuneTrainer(
        midibert,
        train_loader,
        valid_loader,
        test_loader,
        index_layer,
        args.lr,
        args.class_num,
        args.hs,
        y_test.shape,
        args.cpu,
        args.cuda_devices,
        None,
        seq_class,
    )

    print("\nTraining Start")
    save_dir = os.path.join(output_dir, fold_id)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, "model.ckpt")
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    with open(os.path.join(save_dir, "log"), "a") as outfile:
        outfile.write(
            "Loading pre-trained model from " + best_mdl.split("/")[-1] + "\n"
        )
        for epoch in range(args.epochs):
            train_loss, train_acc = trainer.train()
            valid_loss, valid_acc = trainer.valid()
            test_loss, test_acc, _ = trainer.test()

            is_best = valid_acc >= best_acc
            best_acc = max(valid_acc, best_acc)

            if is_best:
                bad_cnt, best_epoch = 0, epoch
            else:
                bad_cnt += 1

            print(
                "epoch: {}/{} | Train Loss: {} | Train F1: {} | Valid Loss: {} | Valid F1: {} | Test loss: {} | Test F1: {}".format(
                    epoch + 1,
                    args.epochs,
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                    test_loss,
                    test_acc,
                )
            )

            trainer.save_checkpoint(
                epoch, train_acc, valid_acc, valid_loss, train_loss, is_best, filename
            )

            outfile.write(
                "Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_F1={}, valid_F1={}, test_F1={}\n".format(
                    epoch + 1,
                    train_loss,
                    valid_loss,
                    test_loss,
                    train_acc,
                    valid_acc,
                    test_acc,
                )
            )


def main():
    # argument
    args = get_args()

    # set seed
    seed = args.seed
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # current gpu
    torch.cuda.manual_seed_all(seed)  # all gpu
    np.random.seed(seed)
    random.seed(seed)

    print("Loading Dictionary")
    with open(args.dict_file, "rb") as f:
        e2w, w2e = pickle.load(f)

    compact_classes = ["Bar", "Position", "Pitch", "Duration"]
    pad_word = [e2w[subclass][f"{subclass} <PAD>"] for subclass in compact_classes]

    folds_lookup = [
        [0, 1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12, 13],
        [14, 15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24, 25],
        [26, 27, 28, 29, 30, 31],
    ]

    mozart_lookup = list(range(35))

    for fold in range(5):
        test_fold = [
            fold,
        ]
        val_fold = [
            (fold + 1) % 5,
        ]
        train_fold = [(fold + 2) % 5, (fold + 3) % 5, (fold + 4) % 5]

        train_indices = []
        valid_indices = []
        test_indices = []

        for cur_fold_id in train_fold:
            for i in range(len(folds_lookup[cur_fold_id])):
                train_indices.append(folds_lookup[cur_fold_id][i])

        for cur_fold_id in val_fold:
            for i in range(len(folds_lookup[cur_fold_id])):
                valid_indices.append(folds_lookup[cur_fold_id][i])

        for cur_fold_id in test_fold:
            for i in range(len(folds_lookup[cur_fold_id])):
                test_indices.append(folds_lookup[cur_fold_id][i])

        print(train_indices)
        print(valid_indices)
        print(test_indices)

        train_model(
            args,
            mozart_lookup,
            train_indices,
            valid_indices,
            test_indices,
            pad_word,
            e2w,
            w2e,
            output_dir=args.output_dir,
            fold_id=str(fold),
        )


if __name__ == "__main__":
    main()
