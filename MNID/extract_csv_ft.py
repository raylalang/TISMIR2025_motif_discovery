"""
[ Melody Extraction ]
Given path to input midi file, save the predicted melody midi file. 
Please note that the model is trained on pop909 dataset (containing 3 classes: melody, bridge, accompaniment), 
so there are 2 interpretations: view `bridge` as `melody` or view it as `accompaniment`.
You could choose the mode - `bridge` is viewed as `melody` by default.

Also, the sequence is zero-padded so that the shape (length) is the same, but it won't affect the results, 
as zero-padded tokens will be excluded in post-processing.
"""

import argparse
import numpy as np
import random
import pickle
import os, csv
import miditoolkit

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertConfig

from melody_extraction.midibert.midi2CP import CP
from melody_extraction.midibert.utils import DEFAULT_VELOCITY_BINS, DEFAULT_FRACTION, DEFAULT_DURATION_BINS, DEFAULT_TEMPO_INTERVALS, DEFAULT_RESOLUTION
from MidiBERT.model import MidiBert
from MidiBERT.finetune_model import TokenClassification


def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--input_dir', type=str, default='../datasets/Beethoven_motif-main/csv_notes_clean'
        , help="directory to the input csv file")
    parser.add_argument('--bps_fh_dir', type=str, default='../datasets/BPS_FH_Dataset')
    parser.add_argument('--output_csv_dir', type=str, required=True, help="directory to save the output csv file")
    parser.add_argument('--dict_file', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--ckpt_dir', type=str, default='')

    ### parameter setting ###
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument('--bridge', default=True, type=boolean_string, help='View bridge as melody (True) or accompaniment (False)')
    
    ### cuda ###
    parser.add_argument('--cpu', action="store_true")  # default: false

    args = parser.parse_args()
    return args


def load_model(args, fold, e2w, w2e):
    print("\nBuilding BERT model")
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                                position_embedding_type='relative_key_query',
                                hidden_size=args.hs)

    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    
    model = TokenClassification(midibert, 4, args.hs)
        
    print('\nLoading ckpt from', args.ckpt_dir)  
    checkpoint = torch.load(os.path.join(args.ckpt_dir, str(fold), 'model_best.ckpt'), map_location='cpu')

    # remove module
    #from collections import OrderedDict
    #new_state_dict = OrderedDict()
    #for k, v in checkpoint['state_dict'].items():
    #    name = k[7:]
    #    new_state_dict[name] = v
    #model.load_state_dict(new_state_dict)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    return model


def inference(model, tokens, pad_CP, device):
    """
        Given `model`, `tokens` (input), `pad_CP` (to indicate which notes are padded)
        Return inference output
    """
    tokens = torch.from_numpy(tokens).to(device)
    pad_CP = torch.tensor(pad_CP).to(device)
    attn = torch.all(tokens != pad_CP, dim=2).float().to(device)

    # forward (input, attn, layer idx)
    pred = model.forward(tokens, attn, -1)                      # pred: (batch, seq_len, class_num)

    # is_valid_note = torch.repeat_interleave(attn.unsqueeze(2), 4, dim=2)
    highest_pred_valid_note = torch.max(torch.where(torch.all(tokens != pad_CP, dim=2), pred[:,:,1] - pred[:,:,3], -100.0))
    highest_pred_valid_note = max(torch.max(torch.where(torch.all(tokens != pad_CP, dim=2), pred[:,:,2] - pred[:,:,3], -100.0))
                                    , highest_pred_valid_note)
    # print (highest_pred_valid_note)

    pred[:,:,1] = pred[:,:,1] - min(highest_pred_valid_note, 0.0)
    pred[:,:,2] = pred[:,:,2] - min(highest_pred_valid_note, 0.0)
    output = np.argmax(pred.cpu().detach().numpy(), axis=-1)   # (batch, seq_len)

    return torch.from_numpy(output), pred.cpu().detach()


def get_melody_events(events, inputs, raw_information, preds, raw_preds, pad_CP, bridge=True):
    """
        Filter out predicted melody events.
        Arguments:
        - events: complete events, including tempo changes and velocity
        - inputs: input compact_CP tokens (batch, seq, CP_class), np.array
        - preds: predicted classes (batch, seq), torch.tensor
            Note for predictions: 1 is melody, 2 is bridge, 3 is piano/accompaniment
        - pad_CP: padded CP representation (list)
        - bridge (bool): whether bridge is viewed as melody
    """
    melody_raw_information = []
    numClass = inputs.shape[-1]
    inputs = inputs.reshape(-1, numClass)
    preds = preds.reshape(-1)

    real_pred = np.argmax(raw_preds.numpy(), axis=-1)
    real_pred = real_pred.reshape(-1)

    pad_CP = np.array(pad_CP)

    # output = np.argmax(pred.cpu().detach().numpy(), axis=-1)

    correct = 0
    total_notes = 0
    confusion = [[0, 0], [0, 0]]

    melody_events = []
    note_ind = 0
    note_raw_information_count = 0
    # print (len(raw_information))
    for event in events:
        if len(event) == 5:     # filter out melody events
            is_melody = real_pred[note_ind] == 1 or (bridge and real_pred[note_ind] == 2)
            is_valid_note = np.all(inputs[note_ind] != pad_CP)
            if is_valid_note and is_melody:
                melody_events.append(event)
                melody_raw_information.append(raw_information[total_notes])

            if is_valid_note:
                if is_melody and raw_information[total_notes][-1] != '':
                    correct += 1
                    confusion[1][1] += 1
                elif (not is_melody) and raw_information[total_notes][-1] == '':
                    correct += 1
                    confusion[0][0] += 1
                elif is_melody and raw_information[total_notes][-1] == '':
                    confusion[0][1] += 1
                elif (not is_melody) and raw_information[total_notes][-1] != '':
                    confusion[1][0] += 1
                total_notes += 1
                # note_raw_information_count = note_raw_information_count + 1
            note_ind += 1
        else:
            melody_events.append(event)

    # for i in range(len(tokens)):
    #     # print (event)
    #     for j in range(len(tokens[i])):
    #         if tokens[i][j][0] != 2:
    #             note_tokens_count = note_tokens_count + 1
    # print (note_tokens_count)
    test_accuracy = (confusion[0][0] + confusion[1][1]) / total_notes
    test_precision = (confusion[1][1]) / max((confusion[0][1] + confusion[1][1]), 1.0)
    test_recall = (confusion[1][1]) / max((confusion[1][0] + confusion[1][1]), 1.0)
    test_f1 = (2.0 * test_precision * test_recall) / max((test_precision + test_recall), 0.0001)

    # print (test_f1, test_accuracy, test_precision, test_recall
    #     , confusion[1][0] + confusion[1][1], np.sum(confusion))
    return melody_events, melody_raw_information, [test_f1, test_accuracy, test_precision, test_recall]


def events2midi(events, output_path, prompt_path=None):
    """
        Given melody events, convert back to midi
    """
    temp_notes, temp_tempos = [], []

    for event in events:
        if len(event) == 1:         # [Bar]
            temp_notes.append('Bar')
            temp_tempos.append('Bar')

        elif len(event) == 5:       # [Bar, Position, Pitch, Duration, Velocity]
            # start time and end time from position
            position = int(event[1].value.split('/')[0]) - 1
            # pitch
            pitch = int(event[2].value)
            # duration
            index = int(event[3].value)
            duration = DEFAULT_DURATION_BINS[index]
            # velocity
            index = int(event[4].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # adding
            temp_notes.append([position, velocity, pitch, duration])

        else:                       # [Position, Tempo Class, Tempo Value]
            position = int(event[0].value.split('/')[0]) - 1
            if event[1].value == 'slow':
                tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(event[2].value)
            elif event[1].value == 'mid':
                tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(event[2].value)
            elif event[1].value == 'fast':
                tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(event[2].value)
            temp_tempos.append([position, tempo])

    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))

    # print (notes[:50])

    # get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
        else:
            position, value = tempo
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            tempos.append([int(st), value])
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION*4*4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
    
    # write  
    midi.dump(output_path)
    # print(f"predicted melody midi file is saved at {output_path}")

    return 

def output_csv_to_file(cleaned_note_list, target_csv_path):
    with open(target_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['onset','midi_number'
            ,'morphetic_number','duration','staff_number','measure','type'])
        for i in range(len(cleaned_note_list)):
            writer.writerow(cleaned_note_list[i])

def main():
    args = get_args()
    
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    compact_classes = ['Bar', 'Position', 'Pitch', 'Duration']
    pad_CP = [e2w[subclass][f"{subclass} <PAD>"] for subclass in compact_classes]

    # preprocess input file
    CP_model = CP(dict=args.dict_file)
    # inference
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("Using", device)
    
    folds_lookup = [
        [0, 1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12, 13],
        [14, 15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24, 25],
        [26, 27, 28, 29, 30, 31]]
    test_5fold_mean_results = np.array([0.0, 0.0, 0.0, 0.0])
    test_5fold_mean_results_list = []

    for fold in range(5):
        # load pre-trained model
        model = load_model(args, fold, e2w, w2e)
        model = model.to(device)
        model.eval()

        test_fold = [fold,]
        test_indices = []

        for cur_fold_id in test_fold:
            for i in range(len(folds_lookup[cur_fold_id])):
                test_indices.append(folds_lookup[cur_fold_id][i] + 1)

        print (test_indices)
        test_mean_results = np.array([0.0, 0.0, 0.0, 0.0])

        # for i in range(1, 33):
        for i in test_indices:
            piece = str(i).zfill(2)
            cur_input_path = os.path.join(args.input_dir, f'{piece}-1.csv')
            cur_db_path = os.path.join(args.bps_fh_dir, str(i), 'dBeats.xlsx')

            if i == 8:
                cur_db_path = cur_db_path.replace('dBeats', 'dbeats')
            
            events, tokens, raw_information = CP_model.prepare_data_from_csv(cur_input_path
                , cur_db_path, args.max_seq_len, offset=0)

            _, tokens2, _ = CP_model.prepare_data_from_csv(cur_input_path
                , cur_db_path, args.max_seq_len, offset=128)

            _, tokens3, _ = CP_model.prepare_data_from_csv(cur_input_path
                , cur_db_path, args.max_seq_len, offset=256)

            # _, tokens4, _ = CP_model.prepare_data_from_csv(cur_input_path
            #     , cur_db_path, args.max_seq_len, offset=384)

            filename = os.path.basename(cur_input_path)
            
            with torch.no_grad():
                predictions, raw_preds = inference(model, tokens, pad_CP, device)
                predictions2, raw_preds2 = inference(model, tokens2, pad_CP, device)
                predictions3, raw_preds3 = inference(model, tokens3, pad_CP, device)
                # predictions4, raw_preds4 = inference(model, tokens4, pad_CP, device)

                # print (raw_preds2.shape)
                raw_preds = torch.reshape(raw_preds, shape=(-1, raw_preds.shape[-1]))
                raw_preds2 = torch.reshape(raw_preds2, shape=(-1, raw_preds2.shape[-1]))
                raw_preds3 = torch.reshape(raw_preds3, shape=(-1, raw_preds3.shape[-1]))
                # raw_preds4 = torch.reshape(raw_preds4, shape=(-1, raw_preds4.shape[-1]))

                pred2_eff = min(raw_preds.shape[0], raw_preds2.shape[0] - 128)
                raw_preds[:pred2_eff] = raw_preds[:pred2_eff] + raw_preds2[128:pred2_eff+128]

                pred3_eff = min(raw_preds.shape[0], raw_preds3.shape[0] - 256)
                raw_preds[:pred3_eff] = raw_preds[:pred3_eff] + raw_preds3[256:pred3_eff+256]

            # post-process    
            melody_events, melody_raw_information, test_results = get_melody_events(events
                , tokens, raw_information, predictions, raw_preds, pad_CP, bridge=args.bridge)

            test_mean_results += np.array(test_results)

            os.makedirs(args.output_csv_dir, exist_ok=True)
            output_csv_to_file(melody_raw_information, os.path.join(args.output_csv_dir, piece+'-1.csv'))

        test_mean_results = test_mean_results / len(test_indices)
        print ('Test mean F1: {:.4f} Acc: {:.4f} P: {:.4f} R: {:.4f}'.format(test_mean_results[0]
                                , test_mean_results[1], test_mean_results[2], test_mean_results[3]))

        test_5fold_mean_results += test_mean_results
        test_5fold_mean_results_list.append(test_mean_results)

    test_5fold_mean_results = test_5fold_mean_results / 5.0
    print ('Test 5-fold overall mean F1: {:.4f} Acc: {:.4f} P: {:.4f} R: {:.4f}'.format(test_5fold_mean_results[0]
                            , test_5fold_mean_results[1], test_5fold_mean_results[2], test_5fold_mean_results[3]))

    test_5fold_mean_results_list = np.array(test_5fold_mean_results_list)
    test_5fold_std = np.std(test_5fold_mean_results_list, axis=0)
    print (test_5fold_mean_results_list.shape, test_5fold_std.shape)

    print ('Std for 5-fold F1: {:.4f} Acc: {:.4f} P: {:.4f} R: {:.4f}'.format(test_5fold_std[0]
                            , test_5fold_std[1], test_5fold_std[2], test_5fold_std[3]))

if __name__ == '__main__':
    main()
