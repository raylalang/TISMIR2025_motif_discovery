# -*- coding: utf-8 -*-
"""
Created on Tue Feb  20  2023
@author: Tsung-Ping Chen
"""
# import time
import numpy as np

np.bool = np.bool_
import os, pickle, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import yaml
import csv
from os.path import join as jpath
from os.path import isdir

# import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path

import pretty_midi
from SIA import *
from mir_eval.pattern import establishment_FPR, occurrence_FPR, three_layer_FPR

"""Baseline algorithms"""
"""https://github.com/wsgan001/repeated_pattern_discovery"""
"""@repeated_pattern_discovery-master"""
import sys

sys.path.insert(0, "./repeated_pattern_discovery")
from dataset import Dataset
from vector import Vector
import new_algorithms
import orig_algorithms
import multiprocessing

jkupdd_corpus = [
    "bachBWV889Fg",
    "beethovenOp2No1Mvt3",
    "chopinOp24No4",
    "gibbonsSilverSwan1612",
    "mozartK282Mvt2",
]
jkupdd_notes_csv = [
    "wtc2f20.csv",
    "sonata01-3.csv",
    "mazurka24-4.csv",
    "silverswan.csv",
    "sonata04-2.csv",
]


def load_all_notes(filename):
    """Load all notes from CSV file"""
    dt = [
        ("onset", np.float32),
        ("pitch", np.int32),
        # ('mPitch', np.int32),
        ("duration", np.float32),
        ("staff", np.int32),
        ("measure", np.int32),
        # ('type', '<U4')
    ]  # datatype

    # Format data as structured array
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=",")
        notes = []
        for row in reader:
            if row[0] != "onset":
                onset = float(int(round(float(row[0]) * 100.0))) / 100
                duration = float(int(round(float(row[3]) * 100.0))) / 100
                note = tuple(
                    [onset, int(row[1]), duration, int(row[4]), int(float(row[5]))]
                )
                notes.append(note)
                # notes.append(tuple([x for i, x in enumerate(row[:6]) if i != 2]))
        notes = np.array(notes, dtype=dt)

    # Get unique notes irrespective of 'staffNum'
    notes = notes[notes["duration"] > 0]
    _, unique_indices = np.unique(notes[["onset", "pitch"]], return_index=True)
    notes = notes[unique_indices]

    return np.sort(notes, order=["onset", "pitch"])


def load_all_motives_csv(filename):
    """Load all motives from CSV file"""
    dt = [
        ("onset", np.float32),
        ("end", np.float32),
        ("type", "<U4"),
        ("measure", np.int32),
        ("start_beat", np.float32),
        ("duration", np.float32),
        ("track", np.int32),
        ("time_sig", "<U5"),
        ("measure_score", np.int32),
        ("onset_midi", np.float32),
        ("end_midi", np.float32),
    ]  # datatype

    # Format data as structured array
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)  # skip header
        motives_csv = np.array([tuple(row) for row in reader], dtype=dt)
    print("number of motives =", motives_csv.size)
    return np.sort(motives_csv, order=["onset", "end"])


def load_all_motives_midi(filename):
    """Load all motives from MIDI file"""
    dt = [
        ("onset", np.float32),
        ("end", np.float32),
        ("pitch", np.int32),
    ]  # datatype

    midi_data = pretty_midi.PrettyMIDI(filename)
    notes_midi = [
        np.array(
            [(note.start, note.end, note.pitch) for note in instrument.notes], dtype=dt
        )
        for instrument in midi_data.instruments
    ]  # [(notes in track i), ...]
    return notes_midi


def load_all_motives(filename_csv, filename_midi):
    """Combine motif informations of CSV and MIDI files"""
    motives_csv = load_all_motives_csv(filename_csv)
    motives_midi = load_all_motives_midi(filename_midi)
    motives = {}
    max_n_notes = 0
    max_duration = 0
    for motif in motives_csv:
        type = motif["type"]
        if type not in motives.keys():
            motives[type] = []  # create a motif type

        track = motif["track"]  # track id
        onset_midi, end_midi = motif[
            ["onset_midi", "end_midi"]
        ]  # onset and end in midi time
        onset_calibration = (
            motif["onset_midi"] - motif["onset"]
        )  # onset calibration if pickup measure
        track_notes = motives_midi[track]
        cond = (track_notes["onset"] >= onset_midi) & (track_notes["onset"] < end_midi)
        motif_notes = track_notes[cond]

        if motif_notes.size > max_n_notes:
            max_n_notes = motif_notes.size
        if motif["end"] - motif["onset"] > max_duration:
            max_duration = motif["end"] - motif["onset"]

        motif_notes["onset"] -= onset_calibration
        motif_notes["end"] -= onset_calibration

        for i in range(len(motif_notes["onset"])):
            motif_notes["onset"][i] = (
                float(int(round(float(motif_notes["onset"][i]) * 100.0))) / 100
            )

        motives[type].append(motif_notes)

    assert (
        len([motif for types in motives.values() for motif in types])
        == motives_csv.size
    )
    print("Piece:", filename_csv)
    print("     number of motif types", len(motives.keys()))
    print("     max_n_notes", max_n_notes)
    print("     max_duration", max_duration)
    return motives


def save_patterns(piece_id, patterns_est, save_dir):
    if save_dir is None:
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    serializable = []
    for motif in patterns_est:
        occs = []
        for occ in motif:
            occs.append([[float(x[0]), int(x[1])] for x in occ])
        serializable.append(occs)
    payload = {"piece": piece_id, "patterns": serializable}
    out_path = Path(save_dir) / f"{piece_id}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)


def _format_metrics(P_est, R_est, F_est, P_occ, R_occ, F_occ, P_thr, R_thr, F_thr):
    return {
        "establishment": {
            "precision": float(P_est),
            "recall": float(R_est),
            "f1": float(F_est),
        },
        "occurrence": {
            "precision": float(P_occ),
            "recall": float(R_occ),
            "f1": float(F_occ),
        },
        "three_layer": {
            "precision": float(P_thr),
            "recall": float(R_thr),
            "f1": float(F_thr),
        },
    }


def save_piece_metrics(metrics_by_piece, save_dir, filename="metrics_by_piece.json"):
    if save_dir is None:
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    payload = {"pieces": metrics_by_piece}
    out_path = Path(save_dir) / filename
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def run_one_data(
    csv_note_dir,
    csv_label_dir,
    motif_midi_dir,
    result_list,
    song_indexes,
    save_dir=None,
):
    for i in song_indexes:
        piece = str(i).zfill(2)
        print("piece", piece)
        filename_notes = os.path.join(csv_note_dir, piece + "-1.csv")
        filename_csv = os.path.join(csv_label_dir, piece + "-1.csv")
        filename_midi = os.path.join(motif_midi_dir, piece + "-1.mid")
        notes = load_all_notes(filename_notes)
        motives = load_all_motives(filename_csv, filename_midi)
        # print (motives)
        # Convert motives to mir_eval format
        patterns_ref = [
            [list(occur[["onset", "pitch"]]) for occur in motif]
            for motif in motives.values()
        ]

        # print (patterns_ref)
        start_time = time.time()
        patterns_est = find_motives(
            notes,
            horizontalTolerance=0,
            verticalTolerance=1,
            adjacentTolerance=(2, 12),
            min_notes=4,
            min_cardinality=0.5,
        )

        avg_duration = 0.0
        for j in range(len(patterns_est)):
            avg_duration = avg_duration + (
                patterns_est[j][0][-1][0] - patterns_est[j][0][0][0]
            )

        avg_duration = avg_duration / len(patterns_est)

        runtime_one = time.time() - start_time
        print("runtime_one %.4f" % runtime_one)
        print("Avg duration %.4f" % avg_duration)

        # Evaluation
        elp = time.time()
        F_est, P_est, R_est = establishment_FPR(patterns_ref, patterns_est)
        F_occ, P_occ, R_occ = occurrence_FPR(patterns_ref, patterns_est)
        F_thr, P_thr, R_thr = three_layer_FPR(patterns_ref, patterns_est)
        print("est P %.4f, R %.4f, F %.4f" % (P_est, R_est, F_est))
        print("occ P %.4f, R %.4f, F %.4f" % (P_occ, R_occ, F_occ))
        print("thr P %.4f, R %.4f, F %.4f" % (P_thr, R_thr, F_thr))
        print("elapsed time, eval %.2f sec" % (time.time() - elp))

        save_patterns(piece + "-1", patterns_est, save_dir)

        result_list[i - 1] = [
            F_est,
            P_est,
            R_est,
            F_occ,
            P_occ,
            R_occ,
            F_thr,
            P_thr,
            R_thr,
            runtime_one,
            len(patterns_est),
            len(patterns_ref),
            avg_duration,
            patterns_est,
        ]


def CSA(pruned_csv_note_dir, csv_label_dir, motif_midi_dir, save_dir=None):
    print("Run CSA algorithm")

    all_P_est, all_R_est, all_F_est = [], [], []
    all_P_occ, all_R_occ, all_F_occ = [], [], []
    all_P_thr, all_R_thr, all_F_thr = [], [], []
    runtime = 0

    thread_num = 16
    song_indexes_list = [[] for i in range(thread_num)]
    for i in range(1, 33):
        song_indexes_list[i % thread_num].append(i)
    print(song_indexes_list)

    manager = multiprocessing.Manager()
    result_list = manager.list([[] for i in range(1, 33)])

    jobs = []
    for j in range(thread_num):
        p = multiprocessing.Process(
            target=run_one_data,
            args=(
                pruned_csv_note_dir,
                csv_label_dir,
                motif_midi_dir,
                result_list,
                song_indexes_list[j],
                save_dir,
            ),
        )
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    print(result_list)

    all_P_est = [result[1] for result in result_list]
    all_R_est = [result[2] for result in result_list]
    all_F_est = [result[0] for result in result_list]
    all_P_occ = [result[4] for result in result_list]
    all_R_occ = [result[5] for result in result_list]
    all_F_occ = [result[3] for result in result_list]
    all_P_thr = [result[7] for result in result_list]
    all_R_thr = [result[8] for result in result_list]
    all_F_thr = [result[6] for result in result_list]

    runtime = sum([result[9] for result in result_list])

    predict_motif_count = [result[10] for result in result_list]
    groundtruth_motif_count = [result[11] for result in result_list]
    avg_duration_list = [result[12] for result in result_list]

    piece_metrics = {}
    for i, result in enumerate(result_list, start=1):
        if not result:
            continue
        piece_id = f"{str(i).zfill(2)}-1"
        piece_metrics[piece_id] = {
            "metrics": _format_metrics(
                result[1],
                result[2],
                result[0],
                result[4],
                result[5],
                result[3],
                result[7],
                result[8],
                result[6],
            ),
            "num_motifs": int(result[10]),
            "num_ref_motifs": int(result[11]),
            "runtime_sec": float(result[9]),
            "avg_predicted_duration": float(result[12]),
        }

    mean_P_est = np.mean(all_P_est)
    mean_R_est = np.mean(all_R_est)
    mean_F_est = np.mean(all_F_est)
    mean_P_occ = np.mean(all_P_occ)
    mean_R_occ = np.mean(all_R_occ)
    mean_F_occ = np.mean(all_F_occ)
    mean_P_thr = np.mean(all_P_thr)
    mean_R_thr = np.mean(all_R_thr)
    mean_F_thr = np.mean(all_F_thr)

    avg_duration = np.mean(avg_duration_list)

    mean_predict_motif_count = np.mean(predict_motif_count)
    mean_groundtruth_motif_count = np.mean(groundtruth_motif_count)
    print("Mean_est P %.4f, R %.4f, F %.4f" % (mean_P_est, mean_R_est, mean_F_est))
    print("Mean_occ P %.4f, R %.4f, F %.4f" % (mean_P_occ, mean_R_occ, mean_F_occ))
    print("Mean_thr P %.4f, R %.4f, F %.4f" % (mean_P_thr, mean_R_thr, mean_F_thr))
    print("Runtime %.4f Averaged Runtime %.4f" % (runtime / 60, runtime / 1920))
    print(
        "Avg GT motif %.4f Avg predicted motif %.4f"
        % (mean_groundtruth_motif_count, mean_predict_motif_count)
    )
    print("Avg motif duration %.4f" % (avg_duration))

    save_piece_metrics(piece_metrics, save_dir)

    # output_dir = 'BPS_MNID_CSA_motif_discovery'
    # os.makedirs(output_dir, exist_ok=True)
    # for i in range(32):
    #     piece = str(i+1)
    #     output_path = os.path.join(output_dir, piece+'-1.pkl')

    #     with open(output_path, 'wb') as f:
    #         pickle.dump(result_list[i][13], f)


def de_vec(vec_obj_list):
    return [tuple(v) for v in vec_obj_list]


def get_all_occurrences(tec):
    return [
        [tuple(point + translator) for point in tec.get_pattern()]
        for translator in tec.get_translators()
    ]


def mtps_to_tecs(mtps, dataset):
    sorted_dataset = Dataset.sort_ascending(dataset)
    v, w = orig_algorithms.compute_vector_tables(sorted_dataset)
    ciss = [
        [sorted_dataset._vectors.index(point) for point in intersection]
        for diff_vec, intersection in mtps
    ]
    mcps = [(mtp[1], cis) for mtp, cis in zip(mtps, ciss)]
    orig_algorithms.remove_trans_eq_mtps(mcps)
    tecs = orig_algorithms.compute_tecs_from_mcps(sorted_dataset, w, mcps)
    return tecs


def SIATEC(pruned_csv_note_dir, csv_label_dir, motif_midi_dir, save_dir=None):
    print("******* SIATEC *******")
    all_P_est, all_R_est, all_F_est = [], [], []
    all_P_occ, all_R_occ, all_F_occ = [], [], []
    all_P_thr, all_R_thr, all_F_thr = [], [], []
    piece_metrics = {}

    runtime = 0.0

    for i in range(1, 33):
        piece = str(i).zfill(2)
        print("piece", piece)

        filename_csv = os.path.join(csv_label_dir, piece + "-1.csv")
        filename_midi = os.path.join(motif_midi_dir, piece + "-1.mid")
        motives = load_all_motives(filename_csv, filename_midi)

        # Convert motives to mir_eval format
        patterns_ref = [
            [list(occur[["onset", "pitch"]]) for occur in motif]
            for motif in motives.values()
        ]

        # Read dataset
        # filename_eval = os.path.join(baseline_note_dir, str(i) + '.csv')
        filename_eval = os.path.join(pruned_csv_note_dir, piece + "-1.csv")
        dataset = Dataset(filename_eval)
        print("len(dataset)", len(dataset))

        # Get all the occurrences of all the maximal repeated patterns in the dataset
        elp = time.time()
        """Baseline algorithms"""
        tecs = new_algorithms.siatechf(dataset, min_cr=2)
        runtime_piece = time.time() - elp
        print("elapsed time, tec %.2f sec" % runtime_piece)

        # Convert tecs to mir_eval format
        patterns_est = [
            get_all_occurrences(tec) for tec in tecs if len(tec.get_translators())
        ]
        print("len(patterns_est)", len(patterns_est))
        runtime = runtime + runtime_piece

        save_patterns(piece + "-1", patterns_est, save_dir)

        # Evaluation
        elp = time.time()
        F_est, P_est, R_est = establishment_FPR(patterns_ref, patterns_est)
        F_occ, P_occ, R_occ = occurrence_FPR(patterns_ref, patterns_est)
        F_thr, P_thr, R_thr = three_layer_FPR(patterns_ref, patterns_est)
        print("est P %.4f, R %.4f, F %.4f" % (P_est, R_est, F_est))
        print("occ P %.4f, R %.4f, F %.4f" % (P_occ, R_occ, F_occ))
        print("thr P %.4f, R %.4f, F %.4f" % (P_thr, R_thr, F_thr))
        print("elapsed time, eval %.2f sec" % (time.time() - elp))

        piece_id = piece + "-1"
        piece_metrics[piece_id] = {
            "metrics": _format_metrics(
                P_est,
                R_est,
                F_est,
                P_occ,
                R_occ,
                F_occ,
                P_thr,
                R_thr,
                F_thr,
            ),
            "num_motifs": len(patterns_est),
            "num_ref_motifs": len(patterns_ref),
            "runtime_sec": float(runtime_piece),
        }

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

    mean_P_est = np.mean(all_P_est)
    mean_R_est = np.mean(all_R_est)
    mean_F_est = np.mean(all_F_est)
    mean_P_occ = np.mean(all_P_occ)
    mean_R_occ = np.mean(all_R_occ)
    mean_F_occ = np.mean(all_F_occ)
    mean_P_thr = np.mean(all_P_thr)
    mean_R_thr = np.mean(all_R_thr)
    mean_F_thr = np.mean(all_F_thr)
    print("Mean_est P %.4f, R %.4f, F %.4f" % (mean_P_est, mean_R_est, mean_F_est))
    print("Mean_occ P %.4f, R %.4f, F %.4f" % (mean_P_occ, mean_R_occ, mean_F_occ))
    print("Mean_thr P %.4f, R %.4f, F %.4f" % (mean_P_thr, mean_R_thr, mean_F_thr))

    print("Runtime %.4f Averaged Runtime %.4f" % (runtime / 60, runtime / 1920))

    save_piece_metrics(piece_metrics, save_dir)


def run_one_data_SIATEC_CS(
    csv_note_dir,
    csv_label_dir,
    motif_midi_dir,
    result_list,
    song_indexes,
    save_dir=None,
):
    for song_id in song_indexes:
        piece = str(song_id).zfill(2)
        print("piece", piece)
        filename_notes = os.path.join(csv_note_dir, piece + "-1.csv")
        filename_csv = os.path.join(csv_label_dir, piece + "-1.csv")
        filename_midi = os.path.join(motif_midi_dir, piece + "-1.mid")
        notes = load_all_notes(filename_notes)
        motives = load_all_motives(filename_csv, filename_midi)
        # print (motives)
        # Convert motives to mir_eval format
        patterns_ref = [
            [list(occur[["onset", "pitch"]]) for occur in motif]
            for motif in motives.values()
        ]

        start_time = time.time()
        filename_eval = os.path.join(csv_note_dir, piece + "-1.csv")
        dataset = Dataset(filename_eval)
        print("len(dataset)", len(dataset))

        # Get all the occurrences of all the maximal repeated patterns in the dataset
        elp = time.time()
        """Baseline algorithms"""
        tecs = orig_algorithms.siatech_compress(dataset)
        print("elapsed time, tec %.2f sec" % (time.time() - elp))

        # Convert tecs to mir_eval format
        patterns_est = [
            get_all_occurrences(tec) for tec in tecs if len(tec.get_translators())
        ]
        # print (patterns_est)
        # print (dataset._vectors)
        for i in range(len(patterns_est)):
            for j in range(len(patterns_est[i])):
                start = min(
                    [patterns_est[i][j][k][0] for k in range(len(patterns_est[i][j]))]
                )
                end = max(
                    [patterns_est[i][j][k][0] for k in range(len(patterns_est[i][j]))]
                )
                new_occ = []
                for k in range(len(dataset._vectors)):
                    if (
                        dataset._vectors[k][0] >= start
                        and dataset._vectors[k][0] <= end
                    ):
                        new_occ.append(
                            tuple((dataset._vectors[k][0], dataset._vectors[k][1]))
                        )
                patterns_est[i][j] = list(new_occ)

        runtime_one = time.time() - start_time
        print("runtime_one %.4f" % runtime_one)

        # Evaluation
        elp = time.time()
        F_est, P_est, R_est = establishment_FPR(patterns_ref, patterns_est)
        F_occ, P_occ, R_occ = occurrence_FPR(patterns_ref, patterns_est)
        F_thr, P_thr, R_thr = three_layer_FPR(patterns_ref, patterns_est)
        print("est P %.4f, R %.4f, F %.4f" % (P_est, R_est, F_est))
        print("occ P %.4f, R %.4f, F %.4f" % (P_occ, R_occ, F_occ))
        print("thr P %.4f, R %.4f, F %.4f" % (P_thr, R_thr, F_thr))
        print("elapsed time, eval %.2f sec" % (time.time() - elp))

        save_patterns(piece + "-1", patterns_est, save_dir)

        result_list[song_id - 1] = [
            F_est,
            P_est,
            R_est,
            F_occ,
            P_occ,
            R_occ,
            F_thr,
            P_thr,
            R_thr,
            runtime_one,
            len(patterns_est),
            len(patterns_ref),
        ]


def SIATEC_CS(pruned_csv_note_dir, csv_label_dir, motif_midi_dir, save_dir=None):
    print("******* SIATEC_CS *******")

    all_P_est, all_R_est, all_F_est = [], [], []
    all_P_occ, all_R_occ, all_F_occ = [], [], []
    all_P_thr, all_R_thr, all_F_thr = [], [], []
    runtime = 0
    thread_num = 16
    song_indexes_list = [[] for i in range(thread_num)]
    for i in range(1, 33):
        song_indexes_list[i % thread_num].append(i)
    print(song_indexes_list)

    manager = multiprocessing.Manager()
    result_list = manager.list([[] for i in range(1, 33)])

    jobs = []
    for j in range(thread_num):
        p = multiprocessing.Process(
            target=run_one_data_SIATEC_CS,
            args=(
                pruned_csv_note_dir,
                csv_label_dir,
                motif_midi_dir,
                result_list,
                song_indexes_list[j],
                save_dir,
            ),
        )
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    print(result_list)

    all_P_est = [result[1] for result in result_list]
    all_R_est = [result[2] for result in result_list]
    all_F_est = [result[0] for result in result_list]
    all_P_occ = [result[4] for result in result_list]
    all_R_occ = [result[5] for result in result_list]
    all_F_occ = [result[3] for result in result_list]
    all_P_thr = [result[7] for result in result_list]
    all_R_thr = [result[8] for result in result_list]
    all_F_thr = [result[6] for result in result_list]

    runtime = sum([result[9] for result in result_list])

    predict_motif_count = [result[10] for result in result_list]
    groundtruth_motif_count = [result[11] for result in result_list]

    piece_metrics = {}
    for i, result in enumerate(result_list, start=1):
        if not result:
            continue
        piece_id = f"{str(i).zfill(2)}-1"
        piece_metrics[piece_id] = {
            "metrics": _format_metrics(
                result[1],
                result[2],
                result[0],
                result[4],
                result[5],
                result[3],
                result[7],
                result[8],
                result[6],
            ),
            "num_motifs": int(result[10]),
            "num_ref_motifs": int(result[11]),
            "runtime_sec": float(result[9]),
        }

    mean_P_est = np.mean(all_P_est)
    mean_R_est = np.mean(all_R_est)
    mean_F_est = np.mean(all_F_est)
    mean_P_occ = np.mean(all_P_occ)
    mean_R_occ = np.mean(all_R_occ)
    mean_F_occ = np.mean(all_F_occ)
    mean_P_thr = np.mean(all_P_thr)
    mean_R_thr = np.mean(all_R_thr)
    mean_F_thr = np.mean(all_F_thr)

    mean_predict_motif_count = np.mean(predict_motif_count)
    mean_groundtruth_motif_count = np.mean(groundtruth_motif_count)
    print("Mean_est P %.4f, R %.4f, F %.4f" % (mean_P_est, mean_R_est, mean_F_est))
    print("Mean_occ P %.4f, R %.4f, F %.4f" % (mean_P_occ, mean_R_occ, mean_F_occ))
    print("Mean_thr P %.4f, R %.4f, F %.4f" % (mean_P_thr, mean_R_thr, mean_F_thr))
    print("Runtime %.4f Averaged Runtime %.4f" % (runtime / 60, runtime / 1920))
    print(
        "Avg GT motif %.4f Avg predicted motif %.4f"
        % (mean_groundtruth_motif_count, mean_predict_motif_count)
    )

    save_piece_metrics(piece_metrics, save_dir)


def _lr_v0_process(
    piece: str,
    pruned_csv_note_dir: str,
    csv_label_dir: str,
    motif_midi_dir: str,
    save_dir: str,
    lr_config_path: str,
):
    """Worker to run LR_V0 on one piece (used for multiprocessing)."""
    from motif_discovery.learned_retrieval.predict import LRConfig, lr_config_from_dict, predict_piece  # type: ignore

    cfg = LRConfig()
    if lr_config_path:
        cfg_path = Path(lr_config_path).expanduser()
        raw = None
        if cfg_path.suffix.lower() == ".json":
            raw = json.loads(cfg_path.read_text())
        else:
            if yaml is None:
                raise ValueError(
                    f"Install PyYAML or provide JSON for lr_config: {cfg_path}"
                )
            raw = yaml.safe_load(cfg_path.read_text())
        if raw is None:
            raise ValueError(f"Failed to parse lr_config at {cfg_path}")
        cfg = lr_config_from_dict(raw)

    filename_csv = os.path.join(csv_label_dir, piece + ".csv")
    filename_midi = os.path.join(motif_midi_dir, piece + ".mid")
    motives = load_all_motives(filename_csv, filename_midi)
    patterns_ref = [
        [list(occur[["onset", "pitch"]]) for occur in motif]
        for motif in motives.values()
    ]

    filename_eval = os.path.join(pruned_csv_note_dir, piece + ".csv")
    notes = load_all_notes(filename_eval)

    start_time = time.time()
    patterns_est = predict_piece(notes, piece, cfg)
    runtime = time.time() - start_time

    save_patterns(piece, patterns_est, save_dir)

    F_est, P_est, R_est = establishment_FPR(patterns_ref, patterns_est)
    F_occ, P_occ, R_occ = occurrence_FPR(patterns_ref, patterns_est)
    F_thr, P_thr, R_thr = three_layer_FPR(patterns_ref, patterns_est)
    return (
        F_est,
        P_est,
        R_est,
        F_occ,
        P_occ,
        R_occ,
        F_thr,
        P_thr,
        R_thr,
        runtime,
        len(patterns_est),
        len(patterns_ref),
    )


def LR_V0(
    pruned_csv_note_dir,
    csv_label_dir,
    motif_midi_dir,
    save_dir=None,
    lr_config_path=None,
    num_workers: int = 1,
):
    """
    Learned retrieval v0: segments -> embeddings -> retrieval -> clustering -> consolidation.
    """
    print("******* LR_V0 *******")
    from motif_discovery.learned_retrieval.predict import (
        LRConfig,
        lr_config_from_dict,
        predict_piece,
    )  # lazy import to avoid extra deps elsewhere

    cfg = LRConfig()
    if lr_config_path:
        cfg_path = Path(lr_config_path).expanduser()
        raw = None
        if cfg_path.suffix.lower() == ".json":
            raw = json.loads(cfg_path.read_text())
        else:
            raw = yaml.safe_load(cfg_path.read_text())
        if raw is None:
            raise ValueError(f"Failed to parse lr_config at {cfg_path}")
        cfg = lr_config_from_dict(raw)

    all_P_est, all_R_est, all_F_est = [], [], []
    all_P_occ, all_R_occ, all_F_occ = [], [], []
    all_P_thr, all_R_thr, all_F_thr = [], [], []
    piece_metrics = {}
    runtime = 0.0

    # Determine worker count (optional)
    pieces = [str(i).zfill(2) + "-1" for i in range(1, 33)]

    if num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for piece in pieces:
                fut = ex.submit(
                    _lr_v0_process,
                    piece,
                    pruned_csv_note_dir,
                    csv_label_dir,
                    motif_midi_dir,
                    save_dir if save_dir else "",
                    lr_config_path if lr_config_path else "",
                )
                tasks.append((piece, fut))
            for piece, fut in tasks:
                res = fut.result()
                (
                    F_est,
                    P_est,
                    R_est,
                    F_occ,
                    P_occ,
                    R_occ,
                    F_thr,
                    P_thr,
                    R_thr,
                    runtime_piece,
                    _,
                    _,
                ) = res
                runtime += runtime_piece
                piece_metrics[piece] = {
                    "metrics": _format_metrics(
                        P_est,
                        R_est,
                        F_est,
                        P_occ,
                        R_occ,
                        F_occ,
                        P_thr,
                        R_thr,
                        F_thr,
                    ),
                    "num_motifs": int(res[10]),
                    "num_ref_motifs": int(res[11]),
                    "runtime_sec": float(runtime_piece),
                }
                all_P_est.append(P_est)
                all_R_est.append(R_est)
                all_F_est.append(F_est)
                all_P_occ.append(P_occ)
                all_R_occ.append(R_occ)
                all_F_occ.append(F_occ)
                all_P_thr.append(P_thr)
                all_R_thr.append(R_thr)
                all_F_thr.append(F_thr)
                print(f"{piece}: est P {P_est:.4f}, R {R_est:.4f}, F {F_est:.4f}")
    else:
        for piece in pieces:
            print("piece", piece)

            filename_csv = os.path.join(csv_label_dir, piece + ".csv")
            filename_midi = os.path.join(motif_midi_dir, piece + ".mid")
            motives = load_all_motives(filename_csv, filename_midi)

            patterns_ref = [
                [list(occur[["onset", "pitch"]]) for occur in motif]
                for motif in motives.values()
            ]

            filename_eval = os.path.join(pruned_csv_note_dir, piece + ".csv")
            notes = load_all_notes(filename_eval)

            start_time = time.time()
            patterns_est = predict_piece(notes, piece, cfg)
            runtime_piece = time.time() - start_time
            runtime += runtime_piece

            save_patterns(piece, patterns_est, save_dir)

            elp = time.time()
            F_est, P_est, R_est = establishment_FPR(patterns_ref, patterns_est)
            F_occ, P_occ, R_occ = occurrence_FPR(patterns_ref, patterns_est)
            F_thr, P_thr, R_thr = three_layer_FPR(patterns_ref, patterns_est)
            print("est P %.4f, R %.4f, F %.4f" % (P_est, R_est, F_est))
            print("occ P %.4f, R %.4f, F %.4f" % (P_occ, R_occ, F_occ))
            print("thr P %.4f, R %.4f, F %.4f" % (P_thr, R_thr, F_thr))
            print("elapsed time, eval %.2f sec" % (time.time() - elp))

            piece_metrics[piece] = {
                "metrics": _format_metrics(
                    P_est,
                    R_est,
                    F_est,
                    P_occ,
                    R_occ,
                    F_occ,
                    P_thr,
                    R_thr,
                    F_thr,
                ),
                "num_motifs": len(patterns_est),
                "num_ref_motifs": len(patterns_ref),
                "runtime_sec": float(runtime_piece),
            }

            all_P_est.append(P_est)
            all_R_est.append(R_est)
            all_F_est.append(F_est)
            all_P_occ.append(P_occ)
            all_R_occ.append(R_occ)
            all_F_occ.append(F_occ)
            all_P_thr.append(P_thr)
            all_R_thr.append(R_thr)
            all_F_thr.append(F_thr)

    mean_P_est = np.mean(all_P_est)
    mean_R_est = np.mean(all_R_est)
    mean_F_est = np.mean(all_F_est)
    mean_P_occ = np.mean(all_P_occ)
    mean_R_occ = np.mean(all_R_occ)
    mean_F_occ = np.mean(all_F_occ)
    mean_P_thr = np.mean(all_P_thr)
    mean_R_thr = np.mean(all_R_thr)
    mean_F_thr = np.mean(all_F_thr)

    print("Mean_est P %.4f, R %.4f, F %.4f" % (mean_P_est, mean_R_est, mean_F_est))
    print("Mean_occ P %.4f, R %.4f, F %.4f" % (mean_P_occ, mean_R_occ, mean_F_occ))
    print("Mean_thr P %.4f, R %.4f, F %.4f" % (mean_P_thr, mean_R_thr, mean_F_thr))
    print("Runtime %.4f Averaged Runtime %.4f" % (runtime / 60, runtime / 1920))

    save_piece_metrics(piece_metrics, save_dir)


def get_args():
    parser = argparse.ArgumentParser(description="Motif Discovery Experiments")
    parser.add_argument("--csv_note_dir", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["CSA", "SIATEC", "SIATEC_CS", "LR_V0", "LR_V1"],
    )
    parser.add_argument(
        "--csv_label_dir",
        type=str,
        default="../datasets/Beethoven_motif-main/csv_label",
    )
    parser.add_argument(
        "--motif_midi_dir",
        type=str,
        default="../datasets/Beethoven_motif-main/motif_midi",
    )
    parser.add_argument(
        "--save_predictions_dir",
        type=str,
        default=None,
        help="Optional directory to save predicted motifs per piece (JSON).",
    )
    parser.add_argument(
        "--lr_config",
        type=str,
        default=None,
        help="Optional JSON/YAML config for LR_V0 parameters.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="For LR_V0: number of processes to use (per-piece parallelism).",
    )
    # parser.add_argument('--jkupdd_data_dir', type=str, default='./JKUPDD/JKUPDD-noAudio-Aug2013/groundTruth')
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    method = args.method

    if method == "CSA":
        CSA(
            args.csv_note_dir,
            args.csv_label_dir,
            args.motif_midi_dir,
            args.save_predictions_dir,
        )
    elif method == "SIATEC":
        SIATEC(
            args.csv_note_dir,
            args.csv_label_dir,
            args.motif_midi_dir,
            args.save_predictions_dir,
        )
    elif method == "SIATEC_CS":
        SIATEC_CS(
            args.csv_note_dir,
            args.csv_label_dir,
            args.motif_midi_dir,
            args.save_predictions_dir,
        )
    elif method == "LR_V0":
        LR_V0(
            args.csv_note_dir,
            args.csv_label_dir,
            args.motif_midi_dir,
            args.save_predictions_dir,
            args.lr_config,
            args.num_workers,
        )
    elif method == "LR_V1":
        LR_V0(
            args.csv_note_dir,
            args.csv_label_dir,
            args.motif_midi_dir,
            args.save_predictions_dir,
            args.lr_config,
            args.num_workers,
        )
    else:
        print("Unknown method:", method)
