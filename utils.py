alphabet = 'abcdefghijklmnopqrstuvwxyz'


def read_find_and_convert_to_np_array(file_name):
    with open(file_name) as fp:
        line = fp.readline()
        result = []
        while line:
            lower_line = line.lower().strip()
            arr = []
            for c in lower_line:
                arr.append(alphabet.index(c))
            result.append(arr)
            line = fp.readline()
        return result


def convert_array_to_str(arr):
    s = ''
    for i in arr:
        s += alphabet[i]
    return s
