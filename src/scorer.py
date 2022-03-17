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
