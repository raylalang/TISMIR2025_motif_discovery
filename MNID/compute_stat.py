import argparse
import numpy as np
import pickle
import os, csv, sys

import random
import xlrd

def extract_events_from_csv(csv_note_path, motif_notes, non_motif_notes):
    # Get notes
    with open(csv_note_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row[0] != 'onset':
                onset = float(int(float(row[0]) * 100.0)) / 100
                duration = float(int(float(row[3]) * 100.0)) / 100

                if duration > 0:
                    if row[-1] == '':
                        non_motif_notes.append(int(row[1]))
                    else:
                        motif_notes.append(int(row[1]))

    return motif_notes, non_motif_notes


motif_notes = []
non_motif_notes = []
for fold_id in range(5):
    for indice in range(35):
        piece = str(indice+1).zfill(2)
        csv_note_path = os.path.join('../mozart_sonata_clean_' + str(fold_id), piece+'-1.csv')
        # downbeat_csv_path = os.path.join('../mozart_sonata_clean_dbeat', piece+'-1.csv')
        downbeat_csv_path = os.path.join('../mozart_sonata_clean_correct_dbeat', piece+'-1.csv')

        motif_notes, non_motif_notes = extract_events_from_csv(csv_note_path, motif_notes, non_motif_notes)

print (len(motif_notes), len(non_motif_notes), np.mean(motif_notes), np.mean(non_motif_notes))

motif_notes = []
non_motif_notes = []
for indice in range(35):
    piece = str(indice+1).zfill(2)
    csv_note_path = os.path.join('../mozart_sonata_clean', piece+'-1.csv')
    # downbeat_csv_path = os.path.join('../mozart_sonata_clean_dbeat', piece+'-1.csv')
    downbeat_csv_path = os.path.join('../mozart_sonata_clean_correct_dbeat', piece+'-1.csv')

    motif_notes, non_motif_notes = extract_events_from_csv(csv_note_path, motif_notes, non_motif_notes)

print (len(motif_notes), len(non_motif_notes), np.mean(motif_notes), np.mean(non_motif_notes))

# motif_notes = []
# non_motif_notes = []
# for indice in range(32):
#     piece = str(indice+1).zfill(2)
#     csv_note_path = os.path.join('../Beethoven_motif-main/csv_notes_clean', piece+'-1.csv')
#     motif_notes, non_motif_notes = extract_events_from_csv(csv_note_path, motif_notes, non_motif_notes)
# print (len(motif_notes), len(non_motif_notes), np.mean(motif_notes), np.mean(non_motif_notes))