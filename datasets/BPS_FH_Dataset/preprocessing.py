import numpy as np
import math
import xlrd
import numpy.lib.recfunctions as rfn
from scipy import stats
import itertools
import os, csv
from bisect import bisect_right

def output_csv_to_file(note_list, additional_annotations, target_csv_path):
    with open(target_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['onset','midi_note_number'
            ,'morphetic_pitch_number','duration','staff_number','measure','localbeat'
            , 'key', 'degree', 'quality', 'inversion', 'roman numeral notation', 'motif_type', 'boundary'
            , 'hnote', 'lnote'])

        for i in range(len(note_list)):
            cur_note = [note_list[i]['onset'], note_list[i]['pitch'], '', note_list[i]['duration']
                , note_list[i]['staff'], note_list[i]['measure']]
            for j in range(len(additional_annotations[i])):
                cur_note.append(additional_annotations[i][j])
            writer.writerow(cur_note)


def get_training_data(label_type=None):
    """
    x is input data, y is label;
    x has the shape [num_sequences, num_steps, feature_size];
    if label_type == 'chord_symbol',
        y has the shape [num_sequences, num_steps];
    if label_type == 'chord_function',
        y has the shape [num_sequences, num_steps],
        and chord functions can be access by y[num_sequences, num_steps][function_name],
        where 'key', 'pri_deg', 'sec_deg', 'quality', 'inversion' are valid function_name
    """

    print("Preprocessing the BPS-FH dataset:")

    if label_type not in ['chord_symbol', 'chord_function']:
        print('LabelTypeError: %s,' % label_type, 'label_type should be \'chord_symbol\' or \'chord_function\'.')
        quit()

    path = os.path.dirname(os.path.abspath(__file__))
    print('load data...')

    # onset,midi_note_number,morphetic_pitch_number,duration,staff_number,measure,localbeat,boundary,key,quality,inversion,pri_deg,sec_deg
    # onset,midi_note_number,morphetic_pitch_number,duration,staff_number,measure,localbeat,key,degree,quality,inversion
    # ,roman numeral notation,pri_deg,sec_deg,motif_type,boundary,hnote,lnote
    dt = [
        ('onset', np.float32),
        ('pitch', np.int32),
        ('duration', np.float32),
        ('staff', np.int32),
        ('measure', np.int32)
    ] # datatype

    pieces = [None for _ in range(32)]
    tdeviation = [None for _ in range(32)] # time deviation
    for i in range(32):
        fileDir = os.path.join(path, str(i+1), "notes.csv")
        # notes = np.genfromtxt(fileDir, delimiter=',', dtype=dt) # read notes from .csv file
        with open(fileDir, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            notes = []
            for row in reader:
                if row[0] != 'onset':
                    onset = float(int(float(row[0]) * 100.0)) / 100
                    duration = float(int(float(row[3]) * 100.0)) / 100
                    note = tuple([onset, int(row[1]), duration, int(row[4]), int(float(row[5]))])
                    notes.append(note)
        notes = np.array(notes, dtype=dt)
        notes = notes[notes['duration'] > 0]
        pieces[i] = notes
        # print (i, len(notes))

    dbeat_labels = [None for _ in range(32)]
    for i in range(32):
        fileDir = os.path.join(path, str(i+1), "dBeats.xlsx")
        if i == 7:
            fileDir = os.path.join(path, str(i+1), "dbeats.xlsx")

        workbook = xlrd.open_workbook(fileDir)
        sheet = workbook.sheet_by_index(0)
        dbeats = []
        for rowx in range(sheet.nrows):
            cols = sheet.row_values(rowx)
            dbeats.append(cols[0])
        dbeat_labels[i] = list(dbeats)

    # print (dbeat_labels[0])


    t = [('onset', 'float'), ('end', 'float'), ('key', '<U10'), ('degree', '<U10')
            , ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U10')] # datatype
    chord_labels = [None for _ in range(32)]
    chord_timing_labels = [None for _ in range(32)]
    for i in range(32):
        fileDir = os.path.join(path, str(i+1), "chords.xlsx")

        workbook = xlrd.open_workbook(fileDir)
        sheet = workbook.sheet_by_index(0)
        chords = []
        chord_timings = []
        for rowx in range(sheet.nrows):
            cols = sheet.row_values(rowx)
            if isinstance(cols[3], float): # if type(degree) == float
                cols[3] = int(cols[3])
            chords.append(tuple(cols))
            chord_timings.append(cols[0])
        chords = np.array(chords, dtype=t) # convert to structured array

        # print (chords)
        chord_labels[i] = chords
        chord_timing_labels[i] = np.array(chord_timings)


    pieces_motif = []
    bps_notes = [None for _ in range(32)]
    for i in range(32):
        fileDir = os.path.join(path, '../Beethoven_motif-main/csv_notes_clean', str(i+1).zfill(2) + '-1.csv')
        with open(fileDir, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            notes = []
            motif = []
            for row in reader:
                if row[0] != 'onset':
                    motif.append(row[6])
                    onset = float(int(float(row[0]) * 100.0)) / 100
                    duration = float(int(float(row[3]) * 100.0)) / 100
                    
                    note = tuple([onset, int(row[1]), duration, int(row[4]), int(float(row[5]))])
                    notes.append(note)

                    # notes.append(tuple([x for i, x in enumerate(row[:6]) if i != 2]))
            pieces_motif.append(motif)
            # print (i, len(motif))
            bps_notes[i] = np.array(notes, dtype=dt)

    bps_notes_other_labels = []
    for i in range(32):
        other_labels = []
        for j in range(len(bps_notes[i])):
            cur_onset = bps_notes[i][j]['onset']
            last_downbeat_id = bisect_right(dbeat_labels[i], cur_onset)
            if last_downbeat_id == 0:
                last_downbeat = dbeat_labels[i][0] - (dbeat_labels[i][1] - dbeat_labels[i][0])
            else:
                last_downbeat = dbeat_labels[i][last_downbeat_id-1]
            cur_localbeat = cur_onset - last_downbeat
            cur_localbeat = float(int(round(float(cur_localbeat) * 100.0))) / 100

            last_chordtime_id = bisect_right(chord_timing_labels[i], cur_onset)
            last_chordtime_id = max(last_chordtime_id-1, 0)
            last_chordtime = chord_timing_labels[i][last_chordtime_id]
            # localbeat,key,degree,quality,inversion
            # roman numeral notation,pri_deg,sec_deg,motif_type,boundary,hnote,lnote
            # t = [('onset', 'float'), ('end', 'float'), ('key', '<U10'), ('degree', '<U10')
            # , ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U10')]
            cur_motif_type = pieces_motif[i][j]
            if cur_motif_type == '':
                cur_motif_type = 'z'
                boundary = 0
            else:
                boundary = 1
            other_labels.append([cur_localbeat, chord_labels[i][last_chordtime_id]['key']
                , chord_labels[i][last_chordtime_id]['degree']
                , chord_labels[i][last_chordtime_id]['quality'], chord_labels[i][last_chordtime_id]['inversion']
                , chord_labels[i][last_chordtime_id]['rchord'], cur_motif_type, boundary, 0, 0])
            # print (cur_onset, last_chordtime, last_chordtime_id)
        bps_notes_other_labels.append(other_labels)

    os.makedirs('../cnn_v2_128/dataset_pianoroll/new_dest', exist_ok=True)
    for i in range(32):
        target_csv_path = os.path.join('../cnn_v2_128/dataset_pianoroll/new_dest', str(i+1).zfill(2) + '-1.csv')
        # print (bps_notes_other_labels[i])
        output_csv_to_file(bps_notes[i], bps_notes_other_labels[i], target_csv_path)

    return sets

if __name__ == '__main__':

    """
    x: the input data with shape = [num_sequences, num_steps, feature_size]
    y: the ground truth with shape = [num_sequences, num_steps]
    label_type: 'chord_symbol' for STL_BLSTM_RNNModel, and 'chord_function' for MTL_BLSTM_RNNModel 
    """
    [x_train, x_valid, x_test, y_train, y_valid, y_test] = get_training_data(label_type='chord_symbol')










