#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:18:47 2023

@author: nbiancac
"""

import sys
import os


class Context:
    def __enter__(self):
        current_dir = os.getcwd()
        sys.path.append(current_dir + '/../../src/')  # adding nmm source directory to syspath

    def __exit__(self, *args):
        pass
