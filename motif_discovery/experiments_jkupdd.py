# -*- coding: utf-8 -*-
"""
Created on Tue Feb  20  2023
@author: Tsung-Ping Chen
"""
# import time
import numpy as np
np.bool = np.bool_
import os, argparse
import csv
from os.path import join as jpath
from os.path import isdir
# import matplotlib.pyplot as plt
import time

import pretty_midi
from SIA import *
from mir_eval.pattern import establishment_FPR, occurrence_FPR, three_layer_FPR

'''Baseline algorithms'''
'''https://github.com/wsgan001/repeated_pattern_discovery'''
'''@repeated_pattern_discovery-master'''
import sys
sys.path.insert(0, './repeated_pattern_discovery')
from dataset import Dataset
from vector import Vector
import new_algorithms
import orig_algorithms
import multiprocessing

'''directory of the JKUPDD dataset'''
jkupdd_data_dir = 'JKUPDD/JKUPDD-noAudio-Aug2013/groundTruth'
jkupdd_corpus = ['bachBWV889Fg', 'beethovenOp2No1Mvt3', 'chopinOp24No4', 'gibbonsSilverSwan1612', 'mozartK282Mvt2']
jkupdd_notes_csv = ['wtc2f20.csv', 'sonata01-3.csv', 'mazurka24-4.csv', 'silverswan.csv', 'sonata04-2.csv']


def load_all_notes(filename):
    '''Load all notes from CSV file'''
    dt = [
        ('onset', np.float32),
        ('pitch', np.int32),
        # ('mPitch', np.int32),
        ('duration', np.float32),
        ('staff', np.int32),
        ('measure', np.int32),
        # ('type', '<U4')
    ] # datatype

    # Format data as structured array
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        notes = []
        for row in reader:
            if row[0] != 'onset':
                onset = float(row[0])
                duration = float(row[3])
                note = tuple([onset, int(row[1]), duration, int(row[4]), int(float(row[5]))])
                notes.append(note)
                # notes.append(tuple([x for i, x in enumerate(row[:6]) if i != 2]))
        notes = np.array(notes, dtype=dt)

    # Get unique notes irrespective of 'staffNum'
    notes = notes[notes['duration'] > 0]
    _, unique_indices = np.unique(notes[['onset', 'pitch']], return_index=True)
    notes = notes[unique_indices]

    return np.sort(notes, order=['onset', 'pitch'])

def load_jkupdd_notes_csv(csv_dir):
    dt = [
        ('onset', np.float32),
        ('pitch', np.int32),
        ('duration', np.float32),
        ('staff', np.int32),
    ] # datatype

    # Format data as structured array
    with open(csv_dir, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        notes = np.array([tuple([float(x) for i, x in enumerate(row) if i != 2]) for row in reader], dtype=dt)

    # Get unique notes irrespective of 'staffNum'
    _, unique_indices = np.unique(notes[['onset', 'pitch']], return_index=True)
    notes = notes[unique_indices]
    print('deleted notes:', [i for i in range(notes.size) if i not in unique_indices])

    notes = notes[notes['duration'] > 0]
    return np.sort(notes, order=['onset', 'pitch'])

def get_all_occurrences(tec):
    return [[tuple(point + translator) for point in tec.get_pattern()] for translator in tec.get_translators()]

def load_jkupdd_patterns_csv(csv_dir, max_note_onset):
    annotators_dir = [jpath(csv_dir, f) for f in os.listdir(csv_dir) if isdir(jpath(csv_dir, f))]
    # print(annotators_dir)
    patterns_dir = [
        jpath(annotator, f) for annotator in annotators_dir for f in os.listdir(annotator) if isdir(jpath(annotator, f))
    ]
    # print(patterns_dir)

    patterns = []
    for pattern_dir in patterns_dir:
        # if 'barlowAndMorgenstern' in pattern_dir.split('/'):
        #     continue
        occurrences_csv = [
            jpath(pattern_dir, 'occurrences/csv', f)
            for f in os.listdir(jpath(pattern_dir, 'occurrences/csv')) if f.endswith('.csv')
        ]
        # print(occurrences_csv)
        pattern = []
        for occurrence_csv in occurrences_csv:
            with open(occurrence_csv, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                cur_occ = [tuple([float(x) for x in row]) for row in reader]
                # print (cur_occ)
                if cur_occ[-1][0] <= max_note_onset:
                    pattern.append(cur_occ)
        # print(pattern)
        if len(pattern) >= 2:
            patterns.append(list(pattern))

    print('number of patterns', len(patterns))
    [print('pattern %s with %d occurrences' % (chr(i+65), len(pattern)))for i, pattern in enumerate(patterns)]
    return patterns


def main(pruned_csv_note_dir, method):

    all_P_est, all_R_est, all_F_est = [], [], []
    all_P_occ, all_R_occ, all_F_occ = [], [], []
    all_P_thr, all_R_thr, all_F_thr = [], [], []
    runtime = 0
    total_n_notes = 0
    total_occurrences = 0

    for song_id in range(5):
        print('file %s' % jkupdd_notes_csv[song_id])
        piece = str(song_id+1).zfill(2)

        filename_notes = os.path.join(pruned_csv_note_dir, piece+'-1.csv')
        notes = load_all_notes(filename_notes)

        pattern_csv_dir = jpath(jkupdd_data_dir, jkupdd_corpus[song_id], 'monophonic/repeatedPatterns')

        note_csv_dir = jpath(jkupdd_data_dir, jkupdd_corpus[song_id], 'polyphonic/csv', jkupdd_notes_csv[song_id])
        original_poly_notes = load_jkupdd_notes_csv(note_csv_dir)

        dataset = Dataset(filename_notes)
        dataset._vectors = [Vector(list(x)[:2]) for x in dataset]

        if song_id == 0 or song_id == 3:
            patterns_ref = load_jkupdd_patterns_csv(pattern_csv_dir
                , max_note_onset=original_poly_notes[-1][0])
        else:
            patterns_ref = load_jkupdd_patterns_csv(pattern_csv_dir
                , max_note_onset=100000)

        total_n_notes += len(notes)

        occurrences = [len(patterns_ref[j]) for j in range(len(patterns_ref))]
        total_occurrences += sum(occurrences)
        # continue

        start_time = time.time()
        if method == 'CSA':
            patterns_est = find_motives(notes, horizontalTolerance=0, verticalTolerance=3
                , adjacentTolerance=(2, 12),
                    min_notes=4, min_cardinality=0.7, n_context=100000) # proposed algorithm
        elif method == 'SIATEC':
            tecs = new_algorithms.siatechf(dataset, min_cr=2)
            patterns_est = [get_all_occurrences(tec) for tec in tecs if len(tec.get_translators())]
        elif method == 'SIATEC_CS':
            tecs = orig_algorithms.siatech_compress(dataset)
            patterns_est = [get_all_occurrences(tec) for tec in tecs if len(tec.get_translators())]

            for i in range(len(patterns_est)):
                for j in range(len(patterns_est[i])):
                    start = min([patterns_est[i][j][k][0] for k in range(len(patterns_est[i][j]))])
                    end = max([patterns_est[i][j][k][0] for k in range(len(patterns_est[i][j]))])
                    new_occ = []
                    for k in range(len(dataset._vectors)):
                        if dataset._vectors[k][0] >= start and dataset._vectors[k][0] <= end:
                            new_occ.append(tuple((dataset._vectors[k][0], dataset._vectors[k][1])))
                    patterns_est[i][j] = list(new_occ)
                    
        elp = time.time() - start_time
        runtime += elp
        print('elapsed time %.4f sec' % elp)
        print('len(patterns_est)', len(patterns_est))
        start_time = time.time()

        # Evaluation
        # elp = time.time()
        F_est, P_est, R_est = establishment_FPR(patterns_ref, patterns_est)
        F_occ, P_occ, R_occ = occurrence_FPR(patterns_ref, patterns_est)
        F_thr, P_thr, R_thr = three_layer_FPR(patterns_ref, patterns_est)
        print('est P %.4f, R %.4f, F %.4f' % (P_est, R_est, F_est))
        print('occ P %.4f, R %.4f, F %.4f' % (P_occ, R_occ, F_occ))
        print('thr P %.4f, R %.4f, F %.4f' % (P_thr, R_thr, F_thr))
        print('elapsed time, eval %.2f sec' % (time.time() - start_time))

        all_P_est.append(P_est)
        all_R_est.append(R_est)
        all_F_est.append(F_est)
        all_P_occ.append(P_occ)
        all_R_occ.append(R_occ)
        all_F_occ.append(F_occ)
        all_P_thr.append(P_thr)
        all_R_thr.append(R_thr)
        all_F_thr.append(F_thr)
        # exit()

    print('avg notes %d' %(total_n_notes/5))
    print ('total occurrences %d' %(total_occurrences))
    mean_P_est = np.mean(all_P_est)
    mean_R_est = np.mean(all_R_est)
    mean_F_est = np.mean(all_F_est)
    mean_P_occ = np.mean(all_P_occ)
    mean_R_occ = np.mean(all_R_occ)
    mean_F_occ = np.mean(all_F_occ)
    mean_P_thr = np.mean(all_P_thr)
    mean_R_thr = np.mean(all_R_thr)
    mean_F_thr = np.mean(all_F_thr)
    print('Mean_est P %.4f, R %.4f, F %.4f' % (mean_P_est, mean_R_est, mean_F_est))
    print('Mean_occ P %.4f, R %.4f, F %.4f' % (mean_P_occ, mean_R_occ, mean_F_occ))
    print('Mean_thr P %.4f, R %.4f, F %.4f' % (mean_P_thr, mean_R_thr, mean_F_thr))
    print('Runtime %.4f min Averaged runtime %.4f min' % (runtime / 60, runtime / 300))

def get_args():
    parser = argparse.ArgumentParser(description='Motif Discovery Experiments')
    parser.add_argument('--csv_note_dir', type=str, required=True)
    parser.add_argument('--method', type=str, required=True, choices=['CSA', 'SIATEC', 'SIATEC_CS'])
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    method = args.method

    main(args.csv_note_dir, method=method)