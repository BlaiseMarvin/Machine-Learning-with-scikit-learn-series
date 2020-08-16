import numpy
from sklearn.datasets import load_breast_cancer
import pandas
cancer=load_breast_cancer()
def answer_one():
    """converts the sklearn 'cancer' bunch

    Returns:
     pandas.DataFrame: cancer data
    """
    data = numpy.c_[cancer.data, cancer.target]
    columns = numpy.append(cancer.feature_names, ["target"])
    return pandas.DataFrame(data, columns=columns)

print(answer_one().shape)
