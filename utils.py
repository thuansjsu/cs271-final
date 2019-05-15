from collections import OrderedDict

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def read_file_and_convert_to_np_array(file_name, append_character=None):
    with open(file_name) as fp:
        line = fp.readline()
        result = []
        count = 1
        while line:
            lower_line = line.lower().strip() if not append_character else line.lower().strip() + append_character
            arr = []
            for c in lower_line:
                arr.append(alphabet.index(c))
            result.append(arr)
            line = fp.readline()
            count += 1
        return result


def convert_array_to_str(arr):
    s = ''
    for i in arr:
        s += alphabet[i]
    return s
