"""
:author: William Coleman
:professor: Prof. Bridget Thomson-McInnes
:class: Introduction to Natural Language Processing
:semester: Spring 2022
:date: 2022.03.10

:about:     Given a tagged dataset, compares the tagged dataset to a key dataset and displays the accuracy and confusion
                matrix for the dataset.


:baseline: 0.8439567788258482

:rule1: The first modification rule is to create a lowercase model and only use it if the word in question is not in the
        primary Most Likely Tag model
:rule1_accuracy: 0.8439567788258482
:rule1_change:  +0.0000000000000000

:rule2: The second modification rule is to create a model that ignores case altogether
:rule2_accuracy: 0.7463747712234268
:rule2_change: -0.0975820076024164

:rule3: The third modification rule is to create a model that explicitly tags all carrdinal numbers using regex
        with a "CD" tag
:rule3_accuracy: 0.8506792904406589
:rule3_change: +0.0067225116148157

:rule4: The fourth modification is to select the default tag to be the most common tag
        in the tagged corpus. THis may or may not be the same as is specified in the rubric.
:rule4_accuracy: 0.8439567788258482
:rule4_change: +0.0000000000000000

:rule5: The fifth modification is to specify two tags that are never correctly tagged and default a guess of those tags ot
        the most-commonly-confused tag
:rule5_accuracy: 0.8439567788258482
:rule5_change: +0.0000000000000000
:rule5_comments: This can absolutely work better, but the examples I chose had too few confusions to have a difference.
    If I were to spend a bit more time performing analysis on the confusion matrix, I could eeasily identify a few more
    promising candidates, but I am out of time for that.



:usage: python3 scorer.py test-data-with-tags.txt test-key.txt
    test-data-with-tags.txt     -   The data tagged by our program
    test-key.txt                -   The key with the correct tags


:example: python3 scorer.py test-data-with-tags.txt test-key.txt
:example_output:
    Accuracy: 0.8506792904406589
    Confusion Matrix:

            LS  PDT   MD  WP$    NN   NNP   )     .  WDT   VBD  #  PRP$  POS    TO    JJ  VBP    $     ,   PRP  [...]
    LS      0    0    0    0     0     0   0     0    0     0  0     0    0     0     0    0    0     0     0
    PDT     0    0    0    0     3     0   0     0    0     0  0     0    0     0     4    0    0     0     0
    [...]
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from common import load_tagged_dataset


class TagTester:
    def __init__(self, test_dataset_path: Path, key_dataset_path: Path):
        self.test_dataset_path = test_dataset_path
        self.key_dataset_path = key_dataset_path
        self.test_dataset = load_tagged_dataset(self.test_dataset_path)
        self.key_dataset = load_tagged_dataset(self.key_dataset_path)
        self.unique_tags = set(x.tag for x in self.test_dataset).union(set(x.tag for x in self.key_dataset))
        self.y_pred = [word.tag for word in self.test_dataset]  # The predicted labels for each word
        self.y_true = [word.tag for word in self.key_dataset]   # The correct labels for each word
        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred, labels=list(self.unique_tags))
        self.accuracy = accuracy_score(self.y_true, self.y_pred)

    def __str__(self) -> str:
        """
        This allows parsing the object directly into a string. The representation of the object will
        print out the accuracy of the model and the confusion matrix.

        :return str: A representation of the test object
        """
        # Parsing the 2D-Arry into a dataframe allows us to easily display the column and indices
        # and to take advantage of the to_string method on the dataframe to parse to tab-delimited csv
        pd_conf_matrix = pd.DataFrame(self.confusion_matrix, columns=self.unique_tags, index=self.unique_tags)
        msg = ""
        msg += f'Accuracy: {self.accuracy}\n\n'
        msg += f'Confusion Matrix:\n{pd_conf_matrix.to_string()}'
        return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='test', type=Path)
    parser.add_argument(dest='key', type=Path)
    args = parser.parse_args()
    tester = TagTester(test_dataset_path=args.test, key_dataset_path=args.key)

    print(tester)


if __name__ == '__main__':
    main()
