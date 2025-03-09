import re
import numpy as np

def decode_to_str(potential_string):
    if isinstance(potential_string, str):
        return potential_string
    elif isinstance(potential_string, bytes):
        return potential_string.decode('ascii') 
    elif isinstance(potential_string, np.ndarray):
        if isinstance(potential_string[0], np.bytes_):
            return ''.join([i.decode('ascii') for i in potential_string])
        elif isinstance(potential_string[0], np.str_):
            return ''.join(potential_string)

def find_substring_between(text, start, end):
    pattern = re.compile(f"{re.escape(start)}(.*?){re.escape(end)}", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1)
    else:
        return None

def array_from_string(array_str):
    array_str_split = array_str.splitlines()
    array = np.zeors((len(array_str_split), len(array_str_split[0])), dtype = bool)
    for i, l in enumerate(array_str.split()):
        if i==0 or i==len(array_str_split):
            continue
        array[i,:] = [int(j) for j in l]
    return array

def find_all_lines_with_substr(text, substring):
    return [line for line in text.splitlines() if substring in line]


def get_target_array(gmoog_script):
    gmoog_script_str = decode_to_str(gmoog_script)
    target_pattern_str = find_substring_between(gmoog_script_str, "targetstart", "targetend")
    return array_from_string(target_pattern_str)

def check_rearrangement_in_master_script(master_script):
    master_script_str = decode_to_str(master_script)
    rearrangement_lines = find_all_lines_with_substr(master_script_str, "call rerng_")
    uncommented_lines = [line for line in rearrangement_lines if not line.startswith("%")]
    return len(uncommented_lines) > 0