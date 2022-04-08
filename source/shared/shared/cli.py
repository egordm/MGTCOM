from simple_parsing import ArgumentParser


def parse_args(*arg_types):
    parser = ArgumentParser()
    for i, arg_type in enumerate(arg_types):
        parser.add_arguments(
            arg_type,
            dest='arg{}'.format(i),
        )
    args = parser.parse_args()
    return [
        getattr(args, 'arg{}'.format(i))
        for i in range(len(arg_types))
    ]
