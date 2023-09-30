# Learn from Your Faults: Leakage Assessment in Fault Attacks Using Deep Learning

## Requirements
- python3

Verified with version 3.10.13.

## Quick Setup
Create a python virtual environment and install all requirements
```
python3 -m venv dl_falat
source dl_falat/bin/activate
pip install -r requirements.txt
```

## Instructions
The method utilizes two main types of inputs: ciphertexts and their corresponding labels, indicating whether they are genuine or faulty.

### Input Data Files
`X.npy`: This is a NumPy array containing 200,000 ciphertexts. It includes both faulty and non-faulty ciphertexts. Each element in this array represents 16 bytes of ciphertexts, aligning with the requirement for byte-level analysis in this PoC.

`y.npy`: This is a NumPy array containing the corresponding labels for the ciphertexts in `X.npy`. A label of '0' denotes that the ciphertext is genuine, while a '1' indicates that it has been generated due to faults.

### Perform the Analysis
Run the following code to generate results for different dataset size.
```
python main.py
```

Run the following code to plot the results.
```
python plot_results.py
```

Once executed, a plot will be generated and saved in the current working directory. You can find it with the filename: `plot.pdf`.

### Analysis Note:
The PoC is designed to perform a byte-level analysis. However, if bit-level analysis is needed, you will have to restructure the `X.npy` file accordingly.

## Cite Us
If you find our work interesting and use it in your research, please cite our paper describing:

Sayandeep Saha, Manaar Alam, Arnab Bag, Debdeep Mukhopadhyay, Pallab Dasgupta, "_Learn from Your Faults: Leakage Assessment in Fault Attacks Using Deep Learning_", Journal of Cryptology, 2023.

### BibTex Citation
```
@article{DBLP:journals/joc/SahaABMD23,
  author       = {Sayandeep Saha and
                  Manaar Alam and
                  Arnab Bag and
                  Debdeep Mukhopadhyay and
                  Pallab Dasgupta},
  title        = {{Learn from Your Faults: Leakage Assessment in Fault Attacks Using
                  Deep Learning}},
  journal      = {Journal of Cryptology},
  volume       = {36},
  number       = {3},
  pages        = {19},
  year         = {2023},
  url          = {https://doi.org/10.1007/s00145-023-09462-6},
  doi          = {10.1007/s00145-023-09462-6},
  timestamp    = {Sat, 20 May 2023 10:53:52 +0200},
  biburl       = {https://dblp.org/rec/journals/joc/SahaABMD23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
## Contact Us
For more information or help with the setup, please contact Manaar Alam at: alam.manaar@nyu.edu
