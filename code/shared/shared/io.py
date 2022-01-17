
def remove_first_line(file_path):
    """
    Remove the first line of a file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    with open(file_path, 'w') as f:
        for line in lines[1:]:
            f.write(line)