# e4040-2021Fall-project
Seed repo for projects for e4040-2021Fall-project
  - Distributed as Github Repo and shared via Github Classroom
  - Contains only README.md file

# The project and how to run code
  - This project is an implemntation of QANet, a model that specifically tackles reading comprehension problems
  - The main notebook should be main.ipynb
  - For the files in layers folder, they are custom layers that are imported in QANet.py, which builds the model and run by train.py
  #### Fisrt, download the datasets from the link 
  - Download SQuAD(v1.1) from: https://deepai.org/dataset/squad 
  - Download GloVe(840B, 300d) from: https://github.com/stanfordnlp/GloVe, and save the file in the path "./orginal_data/"
  #### Second, run preprocess.py to get trainset and devset, both saved in the path "./dataset/"
  #### Third, to run the project one should run train.py

# Organization of this directory
To be populated by students, as shown in previous assignments.
Create a directory/file tree
./
```
├── ECBM4040.2021Fall.QANE.report.jm5134.zo2151.jp4201.pdf
├── QANet.py
├── README.md
├── layers
│   ├── BatchSlice.py
│   ├── ContextQueryAttention.py
│   ├── DepthwiseConv.py
│   ├── LayerDropout.py
│   ├── LayerNormalization.py
│   ├── MultiHeadAttention.py
│   ├── OutputLayer.py
│   ├── PositionEncoding.py
│   └── ZeroPadding.py
├── main.ipynb
├── original_data
├── params.py
├── preprocess.py
├── train.py
└── utils
    ├── __main__.py
    ├── evaluation.py
    ├── output.py
    └── tokenization.py
```
3 directories, 20 files
