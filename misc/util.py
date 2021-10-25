#import re
import os
import sys
import numpy as np


def make_one_hot(label, C=10):
    one_hot = np.zeros(C)
    if label is not None:
        one_hot[label] = 1

    return one_hot


class Struct:
    def __init__(self, **entries):
        rec_entries = {}
        for k, v in entries.items():
            if isinstance(v, dict):
                rv = Struct(**v)
            elif isinstance(v, list):
                rv = []
                for item in v:
                    if isinstance(item, dict):
                        rv.append(Struct(**item))
                    else:
                        rv.append(item)
            else:
                rv = v
            rec_entries[k] = rv
        self.__dict__.update(rec_entries)

    def __str_helper(self, depth):
        lines = []
        for k, v in self.__dict__.items():
            if isinstance(v, Struct):
                v_str = v.__str_helper(depth + 1)
                lines.append("%s:\n%s" % (k, v_str))
            else:
                lines.append("%s: %r" % (k, v))
        indented_lines = ["    " * depth + l for l in lines]
        return "\n".join(indented_lines)

    def __str__(self):
        return "struct {\n%s\n}" % self.__str_helper(1)

    def __repr__(self):
        return "Struct(%r)" % self.__dict__


class Index:
    def __init__(self):
        self.contents = dict()
        self.ordered_contents = []
        self.reverse_contents = dict()

    def __getitem__(self, item):
        if item not in self.contents:
            return None
        return self.contents[item]

    def index(self, item):
        if item not in self.contents:
            idx = len(self.contents) + 1
            self.ordered_contents.append(item)
            self.contents[item] = idx
            self.reverse_contents[idx] = item
        idx = self[item]
        assert idx != 0
        return idx

    def get(self, idx):
        if idx == 0:
            return "*invalid*"
        return self.reverse_contents[idx]

    def __len__(self):
        return len(self.contents) + 1

    def __iter__(self):
        return iter(self.ordered_contents)


'''
For saving outputs to a log file as well as displaying in terminal
Usage:
    output_manager = OutputManager(result_folder)
    ...
    output_manager.say("things to print out {}".format(number_to_print))
'''


class OutputManager(object):
    def __init__(self, result_path, filename='log.txt'):
        self.result_folder = result_path
        self.log_file = open(os.path.join(result_path, filename), 'w')

    def say(self, s):
        self.log_file.write("{}\n".format(s))
        self.log_file.flush()
        sys.stdout.write("{}\n".format(s))
        sys.stdout.flush()


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        print("Creating {}".format(dir_path))
        os.makedirs(dir_path)
    else:
        #raise Exception('Result folder for this experiment already exists')
        pass
