#!/bin/env python
# Last-modified: 06 Dec 2013 03:42:16

from __future__ import absolute_import
from __future__ import print_function
import os

for file in os.listdir("."):
    if file.endswith(".myrun") :
        newfile = file.replace(".myrun", "")
        print("rename " + file + " to " + newfile)
        os.rename(file, newfile)




