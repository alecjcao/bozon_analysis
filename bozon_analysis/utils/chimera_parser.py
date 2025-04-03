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
    
def find_all_lines_with_substr(text, substring):
    return [line for line in text.splitlines() if substring in line]

def array_from_string(array_str):
    array_str_split = array_str.strip().splitlines()
    array = np.zeros((48, 48), dtype = bool)
    for i, l in enumerate(array_str_split):
        l = l.split('%')[0].strip()
        if i==0 or i==len(array_str_split)-1:
            continue
        array[i] = [int(j) for j in l]
    return array

def compress_array(array):
    new_array = np.zeros_like(array)
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array[i,j]:
                new_array[i, 2*(j-24)//3+24] = array[i,j]
    return new_array

def get_target_array(gmoog_script, trigger_count):
    # get base target array
    gmoog_script_str = decode_to_str(gmoog_script)       
    target_pattern_str = find_substring_between(gmoog_script_str, "targetstart", "targetend")
    target_array = array_from_string(target_pattern_str)
    # check for compression
    for line in gmoog_script_str.strip().splitlines():
        line = line.split('%')[0].strip()
        if line.startswith('rearrange'):
            algorithm = line.split()[1] 
        if line.startswith('initx'):
            ncol = line.split()[1].count('1')
        if line.startswith('inity'):
            nrow = line.split()[1].count('1')
    if 'compressx' in algorithm:
        compressed = ('arbssmartscrunchxy' in algorithm and trigger_count >= (nrow+ncol)*3+1+16+1) or \
            (('arbscrunchy' in algorithm) and (trigger_count >= (2*ncol-1)*3+1+16+1))
        if compressed:
            target_array = compress_array(target_array)
    return target_array

def count_commands(script_str, command):
    count = 0
    repeat_stack = [] # keep track of repeats blocks
    for line in script_str.splitlines():
        line = line.split('%')[0].strip() 
        if len(line) == 0:
            continue
        
        # check for repeat start
        repeat_match = re.match(r'repeat: (\w+)', line)
        if repeat_match:
            repeat_num_str = repeat_match.group(1)
            try:
                repeat_num = int(np.round(float(repeat_num_str)))
            except ValueError:
                repeat_num = 1
            repeat_stack.append(repeat_num)
        
        # check for repeat end
        if line == 'end':
            repeat_stack.pop()

        # check for desired command
        if line.startswith(command):
            count_to_add = 1
            for repeat in repeat_stack:
                count_to_add *= repeat
            count += count_to_add
    
    return count

def check_rearrangement_in_master_script(master_script, functions):
    master_script_str = decode_to_str(master_script)
    rearrangement_lines = find_all_lines_with_substr(master_script_str, "call rerng_")
    uncommented_lines = [line for line in rearrangement_lines if not line.startswith("%")]
    rearranged = len(uncommented_lines) > 0
    if rearranged:
        func_name = find_substring_between(uncommented_lines[0], "call ", "()") + '.func'
        rearrange_func_str = decode_to_str(functions[func_name][func_name][:]).strip()
        trigger_count = count_commands(rearrange_func_str, "pulseon: gmmove")
    else:
        trigger_count = 0
    return rearranged, trigger_count