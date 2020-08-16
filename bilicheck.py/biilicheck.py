import numpy
from sklearn.datasets import load_breast_cancer
import pandas
cancer=load_breast_cancer()
def answer_one():
    """converts the sklearn 'cancer' bunch

    Returns:
     pandas.DataFrame: cancer data
    """
    data = numpy.c_[cancer['data'], cancer['target']]
    columns = numpy.append(cancer.feature_names, ["target"])
    return pandas.DataFrame(data, columns=columns)

def answer_two():
    """calculates number of malignent and benign

    Returns:
     pandas.Series: counts of each
    """
    cancerdf = answer_one()
    counts = cancerdf.target.value_counts(ascending=True)
    counts.index = "malignant benign".split()
    return counts