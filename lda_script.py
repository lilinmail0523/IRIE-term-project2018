import subprocess
import sys
import os
from xml_read import read
from lda import *

cur_dir = os.getcwd()


def install():
    os.system('export PYTHONPATH=$' + cur_dir + '/:$PYTHONPATH')

    try:
        import pandas
        import tqdm
        import gensim
        warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
        import inflect

        import nltk
        import contractions
    except:

        subprocess.call([sys.executable, "-m", "pip", "install", "--target",  cur_dir,"gensim"])
        subprocess.call([sys.executable, "-m", "pip", "install",  "--target",  cur_dir,"tqdm"])
        subprocess.call([sys.executable, "-m", "pip", "install",  "--target",  cur_dir,"pandas"])
        subprocess.call([sys.executable, "-m", "pip", "install",  "--target",  cur_dir,"nltk"])
        subprocess.call([sys.executable, "-m", "pip", "install", "--target", cur_dir,"contractions"])
        subprocess.call([sys.executable, "-m", "pip", "install", "--target", cur_dir,"inflect"])


if __name__ == '__main__':
    print("---install package---")
    install()
    from xml_read import read
    from bm25 import *
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    print("---Fetch data from xml----")
    read( ["SemEval2017-task3-English-test-input.xml"], "test_use.csv", 1)
    excu_lda()
