import numpy as np
import os
import sys

if __name__ == "__main__":
    args = sys.argv

    value = 2000.0

    args.remove(args[0])

    if "-v" in args:
        value_idx = args.index("-v") + 1
        value = args[value_idx]
        args.remove("-v")
        args.remove(value)

    files = args

    for filename in files:
        # make copy of original and store alongside
        
        #filename = os.path.basename(sys.argv[0])
        print("filename: {0}".format(filename))
        dir_path = os.path.dirname(os.path.abspath(filename))
        filename2 = os.path.basename(filename)

        backup_name = dir_path + "/" + "Original_" + filename2
        print(backup_name)

        if not os.path.isfile(backup_name):
            os.system("cp {0} {1}".format(filename, backup_name))
        else:
            print("Did not copy original file as backup exists for {0}".format(filename))

        last_line_idx = 0
        for line in open(filename, "r"):
            last_line_idx = int(line.split("\t")[0])

        lines_to_pad = 2000 - last_line_idx
        print(last_line_idx)
        print(lines_to_pad)
        
        f = open(filename, "a")
        for idx in range(lines_to_pad):
            f.write("{0}\t{1}\n".format(last_line_idx + idx, value))
