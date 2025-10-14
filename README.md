#CSU_mzl233

This project is a repository for uploading papers and public data related to the "Power Electronics + AI" research of the New Energy Control and Industrial Intelligence Research Institute, Central South University.

#1. Papers:

Please refer to the "papers" folder. Up to now, all three papers have been published in IEEE Trans. Power Electron.

#2. Partial public datasets:

Please refer to the "dataset/dataset_3ports" folder.

(1) Both png and svg files are visual images of the three-port circuit.

(2) The trueall.csv file is the binary storage format of the corresponding circuit, and its meaning is explained as follows:

Each row represents a piece of circuit data, which can correspond to the png and svg files.

Any row is a 5*7*7 three-dimensional matrix stored in a single row after flattening. Among them, 7*7 represents a 7*7 node adjacency matrix.

5 represents layers: the first layer is the non-connection layer, i.e., set to 1 if there is no connection between nodes i and j; the second layer is the V layer, i.e., set to 1 if nodes i and j are connected through a voltage source;

the third layer is the L layer, i.e., set to 1 if nodes i and j are connected through an inductor; the fourth layer is the S layer, i.e., set to 1 if nodes i and j are connected through a power switch;

the fifth layer is the C layer, i.e., set to 1 if nodes i and j are connected through a capacitor.

#End

