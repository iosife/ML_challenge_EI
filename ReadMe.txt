Contents
--------
"matcher.py": Python scipt.
"matcher.ipynb": A Jupyter notebook corresponding to "matcher.py".
"ReadMe.txt": This file.

Notes
-----
[1] Python version used: Python 3.8.10

[2] Need to have in the working dir the datasets "source1.tsv" and "source2.tsv".

[3] About performance
Using a subset of "source1.tsv" (10K entries) and "source2.tsv"
on an Ubuntu machine (20.04.4 LTS) equiped with (i) Intel Core i3-7100T CPU @ 3.40GHz × 4
and (ii) 12GB RAM, the execution of "matcher.py" took approximately 160 minutes.

[4] Alternative (first) approach
For ID matching only, the following command can be used:
cat source1.tsv | awk -F'\t' '{ printf "grep -w -n %d source2.tsv\n", $1 }' | bash - > matched.txt
utilizing UNIX tools including awk and grep.
It can be regarded as a quick "hack", however, it is quite fast.
