import os
from typing import Any, NewType

from argparse import ArgumentParser

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

class InvalidNetwork(Exception):
    def __init__(self, message: str) -> None:            
       super().__init__(message)
       self.message: str = message
