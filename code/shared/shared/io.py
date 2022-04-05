def remove_first_line(file_path):
    """
    Remove the first line of a file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    with open(file_path, 'w') as f:
        for line in lines[1:]:
            f.write(line)


def mock_cli(parse_fn, **kwargs):
    if len(kwargs) > 0:
        from unittest import mock
        argv = []
        for k, v in kwargs.items():
            argv.append(f'--{k}={v}')

        with mock.patch("sys.argv", argv):
            cli = parse_fn()
    else:
        cli = parse_fn()

    return cli
