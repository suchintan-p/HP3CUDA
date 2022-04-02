from sys import argv

fname = argv[1]

with open(fname) as f:
    lines = f.readlines()

layer = -1
comp_time = 0
cum_comp_time = 0
in_size = 0
out_size = 0
trans_time = 0
thresh = 0
cum_thresh = 0
is_check = False

for line in lines:
    if line.startswith("Layer Number"):
        num = int(line.split(":")[1].strip())
        if num == layer:
            continue
        else:
            if layer != -1:
                # Print previous layer
                print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(layer, in_size, out_size, comp_time, cum_comp_time, trans_time, thresh, cum_thresh, is_check))

            layer = num
            is_check = False

    elif line.startswith("Time Taken compute cumulative"):
        num = float(line.split("=")[1].strip())
        cum_comp_time = num

    elif line.startswith("Time Taken compute"):
        num = float(line.split("=")[1].strip())
        comp_time = num

    elif line.startswith("Input Size"):
        num = int(line.split(":")[1].strip())
        in_size = num

    elif line.startswith("Ouput Size"):
        num = int(line.split(":")[1].strip())
        out_size = num

    elif line.startswith("Time Taken transfer"):
        num = float(line.split("=")[1].strip())
        trans_time = num

    elif line.startswith("Thresh"):
        num = float(line.split("=")[1].strip())
        thresh = num

    elif line.startswith("Cumulative Thresh"):
        num = float(line.split("=")[1].strip())
        cum_thresh = num

    elif line.startswith("CHECKPOINT"):
        is_check = True




# Print previous layer
print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(layer, in_size, out_size, comp_time, cum_comp_time, trans_time, thresh, cum_thresh, is_check))
















