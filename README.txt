The code with the data can be found at https://github.com/maayank/cs7641assignment4
i.e. git clone https://github.com/maayank/cs7641assignment4.git

I implemented the code for the assignment using Python3 and heavy use of mdptoolbox-hiive.

I did modify slightly mdptoolbox-hiive to converge VI and PI differently than the default implementation. As such, I include two files:
1. The contained hiive directory with the modified code is the actual module used by the program
2. mdp.diff - includes the exact differences in the standard diff format

I left the original mdptoolbox-hiive in the requirements so all needed dependencies will be installed.
Python module order of evaluation ensures the above dir is used for the module when running as described below.
The requirements.txt file includes the versions I used - pip3 install -r requirements.txt should install them, as usual.
I've used Python 3.8.9.

The main entry point for my implementation is main.py. Running "python3 main.py" should run all the experiments and create all the png files used for the report.

Many thanks
