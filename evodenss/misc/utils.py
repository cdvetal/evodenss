import os
from argparse import ArgumentParser, _AppendAction
from typing import Any, NamedTuple, NewType, no_type_check

InputLayerId = NewType('InputLayerId', int)
LayerId = NewType('LayerId', int)

def is_valid_file(parser: ArgumentParser, arg: Any) -> object:
    if not os.path.isfile(arg):
        parser.error(f"The file {arg} does not exist!")
    else:
        return arg

def is_yaml_file(parser: ArgumentParser, arg: Any) -> object:
    if is_valid_file(parser, arg):
        if not arg.endswith(".yaml"):
            parser.error(f"The file {arg} is not a yaml file")
        else:
            return arg
    parser.error(f"The file {arg} is not a yaml file")


class ConfigPair(NamedTuple):
    key: str
    value: Any


class ConfigPairAction(_AppendAction):

    @no_type_check
    def _copy_items(items):
        if items is None:
            return []
        # The copy module is used only in the 'append' and 'append_const'
        # actions, and it is needed only when the default value isn't a list.
        # Delay its import for speeding up the common case.
        if isinstance(items, list):
            return items[:]
        import copy
        return copy.copy(items)

    @no_type_check
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        if nargs != 2:
            raise ValueError(f"ConfigPairAction requires two args per flag. Current nargs = {nargs}")
        super(_AppendAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    @no_type_check
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = ConfigPairAction._copy_items(items)
        items.append(ConfigPair(*values))
        setattr(namespace, self.dest, items)


class InvalidNetwork(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message
