import src.milestone1 as ms1
import matplotlib.pyplot as plt
import re
import os
import numpy as np


def words_generator(file_obj):
    for line in file_obj:
        for word in re.findall(r"[\w]+", line):
            yield word

def count_word_frequency(fnames):
    word_count_dict = {}
    for file in fnames:
        f = open(file,"r")
        words = words_generator(f)
        for word in words:
            if word not in word_count_dict:
                  word_count_dict[word] = 0
            word_count_dict[word] += 1
    return word_count_dict

def count_glyph_frequency(fnames):
    glyph_count_dict = {}
    for file in fnames:
        f = open(file,"r")
        for line in f:
            for glyph in line:
                if glyph not in glyph_count_dict:
                  glyph_count_dict[glyph] = 0
                glyph_count_dict[glyph] += 1

    return glyph_count_dict

def sort_freq_dict(freq_dict):
    aux = [(freq_dict[key], key) for key in freq_dict]
    aux.sort()
    aux.reverse()
    return aux


if __name__ == '__main__':
    #Option 1: Load 100% of the data and write the words or glyphs to a file
    train, _ = ms1.load_data('GT4HistOCR', n_train=1.0, n_test=0.0)
    #dict = count_glyph_frequency(train.gt_paths)
    #with open('../../glyph_freq.txt', 'w') as file:
    #    file.write(str(dict))
    #Option 2: Open an existing file containing the dictionary
    #dict = eval(open('../../glyph_freq.txt', 'r').read())

    #word_tuple = sort_freq_dict(dict)



