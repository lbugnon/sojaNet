# ======================================================
# Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
# sinc(i) - http://sinc.unl.edu.ar/
# ======================================================


def load_config(filename):
    config = {}
    for line in open(filename):
        if line[0] == "#":
            continue
        
        line = line.strip().split("=")
        if len(line) != 2:
            continue

        val = line[1]

        comment_ind = val.find("#")
        if comment_ind != -1:
            val=val[:comment_ind]
        config[line[0]] = val

    return config


def save_config(config, filename):

    with open(filename, "w") as fout:
        for k in sorted(config.keys()):
            fout.write("%s=%s\n" % (k, config[k]))
