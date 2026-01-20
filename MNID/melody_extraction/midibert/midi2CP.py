import numpy as np
import pickle
from tqdm import tqdm
import melody_extraction.midibert.utils as utils
import csv
from .utils import *
from pathlib import Path
import openpyxl
from fractions import Fraction


class CP(object):
    def __init__(self, dict):
        # load dictionary
        self.event2word, self.word2event = pickle.load(open(dict, "rb"))
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        classes = ["Bar", "Position", "Pitch", "Duration"]
        self.pad_word = [
            self.event2word[etype]["%s <PAD>" % etype] for etype in classes
        ]

    def extract_events_from_csv(self, csv_note_path, downbeat_csv_path):

        # Get notes
        raw_information = []
        notes = []
        with open(csv_note_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                if row[0] != "onset":
                    onset = float(int(float(row[0]) * 100.0)) / 100
                    duration = float(int(float(row[3]) * 100.0)) / 100
                    note = [onset, int(row[1]), duration]
                    if duration > 0:
                        notes.append(note)
                        raw_information.append(row)

        # print (len(notes), len(raw_information))
        note_stretch = 1
        # Get downbeats
        dbeats = [0.0]
        suffix = Path(downbeat_csv_path).suffix.lower()

        if suffix == ".xlsx":
            wb = openpyxl.load_workbook(
                downbeat_csv_path, read_only=True, data_only=True
            )
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

        dbeats_stretch = []
        for i in range(len(dbeats) - 1):
            dbeats_stretch.append(dbeats[i] * note_stretch)
            for j in range(1, note_stretch):
                dbeats_stretch.append(
                    dbeats[i] * note_stretch + j * (dbeats[i + 1] - dbeats[i])
                )

        dbeats_stretch.append(dbeats[-1] * note_stretch)
        dbeats = list(dbeats_stretch)

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
                    # if len(groups) == 2:
                    #     print (cur_downbeat_timing + start_interp_in_bar * (DEFAULT_RESOLUTION * 4)
                    #     , cur_downbeat_timing + end_interp_in_bar * (DEFAULT_RESOLUTION * 4)
                    #     , notes[i][1], db1, db2
                    #     , notes[i])
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
        # print (groups[1:4])
        # print (note_count)
        events = utils.item2event(groups)
        return events, raw_information

    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        # print (note_items)
        if len(note_items) == 0:  # if the midi contains nothing
            return None
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events

    def extract_events_from_csv_beat(self, csv_note_path):

        # Get notes
        dbeats = []
        raw_information = []
        notes = []
        with open(csv_note_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                if row[0] != "onset":
                    dbeats.append(-float(Fraction(row[-1])))
                    dbeats.append(0)
                    break

            for row in reader:
                if row[0] != "onset":
                    onset = float(int(float(row[0]) * 100.0)) / 100
                    duration = float(int(float(row[3]) * 100.0)) / 100
                    note = [onset, max(min(int(row[1]), 107), 24), duration]
                    if duration > 0:
                        notes.append(note)
                        raw_information.append(row)

                    while dbeats[-1] < onset:
                        dbeats.append(dbeats[-1] + float(Fraction(row[-1])))

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
                    # if len(groups) == 2:
                    #     print (cur_downbeat_timing + start_interp_in_bar * (DEFAULT_RESOLUTION * 4)
                    #     , cur_downbeat_timing + end_interp_in_bar * (DEFAULT_RESOLUTION * 4)
                    #     , notes[i][1], db1, db2
                    #     , notes[i])
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
        # print (groups[1:4])
        # print (note_count)
        events = utils.item2event(groups)
        return events, raw_information

    def padding(self, data, max_len):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            data.append(self.pad_word)

        return data

    def prepare_data_from_csv_beat(self, csv_note_path, max_len, offset=0):
        """
        Prepare data for a single midi
        """
        # extract events
        events, raw_information = self.extract_events_from_csv_beat(csv_note_path)
        if not events:  # if midi contains nothing
            raise ValueError(f"The given {csv_note_path} is empty")

        words = []
        for tup in events:
            nts = []
            if len(tup) == 5:  # Note
                for e in tup:
                    if e.name == "Velocity":
                        continue
                    e_text = "{} {}".format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                words.append(nts)

        slice_words = []
        for i in range(-offset, len(words), max_len):
            if i < 0:
                data = []
                pad_len = -i
                for _ in range(pad_len):
                    data.append(self.pad_word)
                for j in range(min(i + max_len, len(words))):
                    data.append(words[j])
                slice_words.append(data)
            else:
                slice_words.append(words[i : i + max_len])

        # padding or drop
        if len(slice_words[-1]) < max_len:
            slice_words[-1] = self.padding(slice_words[-1], max_len)

        slice_words = np.array(slice_words)

        return events, slice_words, raw_information

    def prepare_data_from_csv(
        self, csv_note_path, downbeat_csv_path, max_len, offset=0
    ):
        """
        Prepare data for a single midi
        """
        # extract events
        events, raw_information = self.extract_events_from_csv(
            csv_note_path, downbeat_csv_path
        )
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
                    nts.append(self.event2word[e.name][e_text])
                words.append(nts)

        # slice to chunks so that max length = max_len (default: 512)
        # slice_words = []
        # for i in range(0, len(words), max_len):
        #     slice_words.append(words[i:i+max_len])

        # print (offset, len(words), max_len, len(raw_information))

        slice_words = []
        for i in range(-offset, len(words), max_len):
            if i < 0:
                data = []
                pad_len = -i
                for _ in range(pad_len):
                    data.append(self.pad_word)
                for j in range(min(i + max_len, len(words))):
                    data.append(words[j])
                slice_words.append(data)
            else:
                slice_words.append(words[i : i + max_len])

        # padding or drop
        if len(slice_words[-1]) < max_len:
            slice_words[-1] = self.padding(slice_words[-1], max_len)

        slice_words = np.array(slice_words)
        # print (len(slice_words))

        return events, slice_words, raw_information

    def prepare_data(self, midi_path, max_len):
        """
        Prepare data for a single midi
        """
        # extract events
        events = self.extract_events(midi_path)
        if not events:  # if midi contains nothing
            raise ValueError(f"The given {midi_path} is empty")

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
                    nts.append(self.event2word[e.name][e_text])
                words.append(nts)

        # slice to chunks so that max length = max_len (default: 512)
        slice_words = []
        for i in range(0, len(words), max_len):
            slice_words.append(words[i : i + max_len])

        # padding or drop
        if len(slice_words[-1]) < max_len:
            slice_words[-1] = self.padding(slice_words[-1], max_len)

        slice_words = np.array(slice_words)

        return events, slice_words
