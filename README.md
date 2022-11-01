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

  
# Detailed instructions how to submit this assignment/homework/project:
1. The assignment will be distributed as a github classroom assignment - as a special repository accessed through a link

2. A students copy of the assignment gets created automatically with a special name
3. Students must rename the repo per instructions below

4. The solution(s) to the assignment have to be submitted inside that repository as a set of "solved" Jupyter Notebook(s), and several modified python files which reside in directories/subdirectories

5. Three files/screenshots need to be uploaded into the directory "figures" which prove that the assignment has been done in the cloud

6. Code to be graded from github

7. If some model is too large for github- create google (liondrive) directory, upload, and share the link with E4040TAs@columbia.edu

8. Submit report as a pdf file, through courseworks upload, and also have the pdf report in the root of the github repo for the group


## (Re)naming of a project repository shared by multiple students (TODO students)
INSTRUCTIONS for naming the students' solution repository for assignments with several students, such as the final project. Students must use a 4-letter groupID, the same one that was chosed in the class spreadsheet: 
* Template: e4040-2021Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4040-2021Fall-Project-MEME-zz9999-aa9999-aa0000.

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
