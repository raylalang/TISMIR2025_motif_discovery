# translate and split
import pandas as pd
import os

tonic_translation_dict_major = {'C-':'B', 'D-':'C+', 'E-':'D+', 'E+':'F', 'F-':'E', 'G-':'F+', 'A-':'G+', 'B-':'A+', 'B+':'C'}
tonic_translation_dict_minor = {'c-':'b', 'd-':'c+', 'e-':'d+', 'e+':'f', 'f-':'e', 'g-':'f+', 'a-':'g+', 'b-':'a+', 'b+':'c'}

# Merge both translation dictionaries
tonic_translation_dict = {**tonic_translation_dict_major, **tonic_translation_dict_minor}

tonic_to_number = {'C': 0, 'C+': 1, 'D': 2, 'D+': 3, 'E': 4, 'F': 5, 'F+': 6, 'G': 7, 'G+': 8, 'A': 9, 'A+': 10, 'B': 11,
                   'c': 12, 'c+': 13, 'd': 14, 'd+': 15, 'e': 16, 'f': 17, 'f+': 18, 'g': 19, 'g+': 20, 'a': 21, 'a+': 22, 'b': 23}
quality_to_number = {'M': 0, 'm': 1, 'd': 2, 'a': 3, 'M7': 4, 'm7': 5, 'D7': 6, 'd7': 7, 'h7': 8, 'a6': 9}

# Define the new column order
new_column_order = [
    'onset', 'midi_note_number', 'morphetic_pitch_number', 'duration', 'staff_number', 
    'measure', 'localbeat', 'key', 'degree', 'quality', 'inversion', 
    'roman numeral notation', 'pri_deg', 'sec_deg', 'motif_type', 'boundary', 'hnote', 'lnote'
]

def preprocess_csv(csv_path):
    df = pd.read_csv(csv_path, encoding='latin1')

    # Translate 'key' column
    df['key'] = df['key'].apply(lambda x: x.capitalize() if x in tonic_to_number else x)
    df['key'] = df['key'].apply(lambda x: tonic_to_number.get(tonic_translation_dict.get(x, x), x))

    # Translate 'quality' column
    df['quality'] = df['quality'].apply(lambda x: quality_to_number.get(x, x))

    # Extract relevant columns for degree translation
    df['pri_deg'], df['sec_deg'] = zip(*df['degree'].apply(translate_degree_to_columns))

    # Reorder the columns
    df = df[new_column_order]

    # Print pri_deg and sec_deg values
    # print("pri_deg values:", df['pri_deg'].tolist())
    # print("sec_deg values:", df['sec_deg'].tolist())

    return df


def translate_degree(degree_str):
    if ('+' not in degree_str and '-' not in degree_str) or ('+' in degree_str and degree_str[1] == '+'):
        degree = int(degree_str[0])
    elif degree_str[0] == '-':
        degree = int(degree_str[1]) + 14
    elif degree_str[0] == '+':
        degree = int(degree_str[1]) + 7
    else:
        # Default value if none of the conditions match
        degree = 0

    return degree


def translate_degree_to_columns(degree): # (7 diatonics *  3 chromatics  = 21: {0-6 diatonic, 7-13 sharp, 14-20 flat})
    if '/' not in degree:
        pri_deg = translate_degree(degree)
        sec_deg = 1
    else:
        pri_deg_str, sec_deg_str = degree.split('/')
        pri_deg = translate_degree(pri_deg_str)
        sec_deg = translate_degree(sec_deg_str)

    return pri_deg-1, sec_deg-1 #-1 for 0-20 (21)




def process_all_csvs(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            input_csv_path = os.path.join(input_directory, filename)
            df = preprocess_csv(input_csv_path)

            # Save the revised DataFrame to a new CSV file in the output directory
            output_csv_path = os.path.join(output_directory, f"{filename}") #revised_
            df.to_csv(output_csv_path, index=False)

# Example usage
# input_csv_directory = './src/motif_new_annotataion_dataset' 
# output_csv_directory = './src/motif_new_annotataion_dataset_preprocess'

input_csv_directory = '../cnn_v2_128/dataset_pianoroll/new_dest' 
output_csv_directory = '../cnn_v2_128/dataset_pianoroll/new_dest_preprocessed'
os.makedirs(output_csv_directory, exist_ok=True)

# Process all CSV files in the input directory
process_all_csvs(input_csv_directory, output_csv_directory)

# Process all CSV files in the input directory
# process_all_csvs(input_csv_directory, output_csv_directory)
print("done")




#-------------------------------------------------------------------
# #test_sequence_key
# tonic_translation_dict_major = {'C-':'B', 'D-':'C+', 'E-':'D+', 'E+':'F', 'F-':'E', 'G-':'F+', 'A-':'G+', 'B-':'A+', 'B+':'C'}
# tonic_translation_dict_minor = {'c-':'b', 'd-':'c+', 'e-':'d+', 'e+':'f', 'f-':'e', 'g-':'f+', 'a-':'g+', 'b-':'a+', 'b+':'c'}

# # Merge both translation dictionaries
# tonic_translation_dict = {**tonic_translation_dict_major, **tonic_translation_dict_minor}

# tonic_to_number = {'C': 0, 'C+': 1, 'D': 2, 'D+': 3, 'E': 4, 'F': 5, 'F+': 6, 'G': 7, 'G+': 8, 'A': 9, 'A+': 10, 'B': 11,
#                    'c': 12, 'c+': 13, 'd': 14, 'd+': 15, 'e': 16, 'f': 17, 'f+': 18, 'g': 19, 'g+': 20, 'a': 21, 'a+': 22, 'b': 23}

# # Given sequence
# sequence = ['f', 'f', 'f', 'A-', 'A-', 'A-', 'c+', 'c+', 'D', 'D']

# # Convert the sequence following the rules
# converted_sequence = [tonic_to_number[tonic_translation_dict[tonic.capitalize()]] if tonic.capitalize() in tonic_translation_dict else tonic_to_number.get(tonic, tonic) for tonic in sequence]

# print(converted_sequence)

#-------------------------------------------------------------------
