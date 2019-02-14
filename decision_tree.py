"""
CAPP30122 W'19: Building decision trees

Ziyu Ye, Kunyu He
"""

import math
import sys
import pandas as pd
import numpy as np


class DecisionTree:
    """
    The data structure that can represent a tree for decision making.

    See constructor for detailed description of attributes.
    """

    def __init__(self, data, attributes):
        """
        Construct a DecisionTree object.

        Parameters:
            - data (DataFrame): raw data to be structured
            - attributes (list of strings): the feature names of the data

        Attributes:
            - data (DataFrame): raw data to be structured
            - attributes (list of strings): a sorted list of feature names
            - target (str): name of the last column (target) of the data
            - label (str): value from the target column that occurs most often
            - split_attr (str): column that data at the node to be split on
            - children (dict): a dictionary mapping edges (certain values of
                               attributes) to DecisionTree objects
        """
        self.data = data
        self.attributes = sorted(attributes)
        self.__target = self.data.columns[-1]

        values = sorted(self.data[self.__target].unique())
        self.__label = values[np.argmax([sum(self.data[self.__target] == value)
                                         for value in values])]
        self.__split_attr = None
        self.__children = {}

    @staticmethod
    def __rate(data, attribute, value):
        """
        Calculate the proportion of observations in a subset where an
        attribute is equal to certain value.

        Inputs:
            - data (DataFrame): the data set
            - attribute (str): name of a column in the data set
            - value (any): the value corresponding to the attribute

        Returns:
            - (float)
        """
        return sum(data[attribute] == value) / data.shape[0]

    def __attr_gini(self, data, attribute):
        """
        Compute the gini coefficient of an attribute in the data set.

        Inputs:
            - data (DataFrame): the data set
            - attribute (str): name of a column in the data set

        Returns:
            - (float)
        """
        gini = 1
        for value in data[attribute].unique():
            gini -= self.__rate(data, attribute, value)**2
        return gini

    def __attr_gain_ratio(self, attribute):
        """
        Calculate gain ratio of splitting the data set on a given column.

        Inputs:
            - attribute (str): name of the column to split on

        Returns:
            - (float)
        """
        gain = self.__attr_gini(self.data, self.__target)
        split_info = 0
        attr = self.data[attribute]

        for value in attr.unique():
            gain -= self.__rate(self.data, attribute, value) * \
                    self.__attr_gini(self.data[attr == value], self.__target)
            split_info -= self.__rate(self.data, attribute, value) * \
                          math.log(self.__rate(self.data, attribute, value))

        if split_info == 0:
            return 0
        return gain / split_info

    def find_best_split(self):
        """
        Find the attribute with the largest gain ratio and set it as the
        attribute for the decision tree to split on.

        Inputs:
            - None (the DecisionTree itself)

        Returns:
            - (None)
        """
        self.__split_attr = self.attributes[np.argmax([
            self.__attr_gain_ratio(attr) for attr in self.attributes])]

    def is_leaf(self):
        """
        Check whether the current tree is actually a leaf node. It is a leaf
        node when target attribute in the tree data takes only one value, or
        the attributes is an empty set, or observations take identical value
        across all the columns.

        Inputs:
            - None (the Decision Tree itself)

        Returns:
            - (Boolean) True if it is a leaf node; otherwise False
        """
        if any([self.data[self.__target].nunique() == 1, not self.attributes,
                all(self.data[self.attributes].apply(lambda col:
                                                     col.nunique() == 1))]):
            return True
        return False

    def train(self):
        """
        Build the decision tree recursively based on the given training data.

        Inputs:
            - None (the Decision Tree itself)

        Returns:
            - (Decision Tree) a trained DecisionTree object
        """
        if self.is_leaf():
            return self

        self.find_best_split()
        if self.__attr_gain_ratio(self.__split_attr) == 0:
            return self

        for edge in self.data[self.__split_attr].unique():
            sub_data = self.data[self.data[self.__split_attr] == edge]
            sub_attr = list(filter(lambda x: x != self.__split_attr,
                                   self.attributes))
            self.__children[edge] = DecisionTree(sub_data, sub_attr).train()
        return self

    def classify(self, row):
        """
        Classify a certain row from the test set based on the trained tree and
        return the label iteratively.

        Inputs:
            - row (numpy Series): a row from the test set

        Returns:
            - (str) label of the input row
        """
        if not self.__children or \
                row[self.__split_attr] not in self.__children:
            return self.__label

        return self.__children[row[self.__split_attr]].classify(row)


def go(training_filename, testing_filename):
    """
    Construct a decision tree using the training data and then apply
    it to the testing data.

    Inputs:
      - training_filename (str): the name (with path) of the file with the
                                 training data
      - testing_filename (str): the name (with path) of the file with the
                                testing data

    Returns:
       - output (list of strings): result of applying the decision tree to the
                                   testing data
    """
    train = pd.read_csv(training_filename, dtype=str)
    test = pd.read_csv(testing_filename, dtype=str)
    output = []

    trained_tree = DecisionTree(train, list(train.columns[:-1])).train()
    for _, row in test.iterrows():
        output.append(trained_tree.classify(row))

    return output


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python3 {} <training filename> <testing filename>".format(
            sys.argv[0]))
        sys.exit(1)

    for result in go(sys.argv[1], sys.argv[2]):
        print(result)
