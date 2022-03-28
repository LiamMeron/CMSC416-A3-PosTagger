"""
:author: William Coleman
:professor: Prof. Bridget Thomson-McInnes
:class: Introduction to Natural Language Processing
:semester: Spring 2022
:date: 2022.03.10

:about:     Creates a most likely tag POS tagger on a corpus and tags a test dataset with the model. Prints the tagged dataset
        to the console. The dataset is tagged in the format WORD/TAG. If the word contains a "/" character it will be escaped
        as "\/"


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



:usage: python3 tagger.py corpus.txt body-to-tag.txt [-r {1-5}]
    corpus.txt      -   The tagged input file to train on
    body-to-tag.txt -   The text that needs tagging
    -r [Optional]   -   Which model to use. Each of the 5 models can be run by specifying the version to use.
                            Defaults to baseline tagger

:example: python3 tagger.py pos-test-with-tags.py pos-test-key.txt
:example_output:

"""

import argparse
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from common import TaggedWord, load_tagged_dataset


class Tagger:
    def __init__(self, dataset_path: Path, default_tag: str = 'NN', rule: int = None):
        self._DEFAULT_TAG = default_tag

        self.dataset_path = dataset_path
        self.dataset = load_tagged_dataset(self.dataset_path)

        if rule == 1:
            """Rule 1 uses baseline model, and if word is not present in baseline model, then uses lowercase model 
            with default tag """
            self.train = self.train_baseline
            self.baseline_model = self.train()
            self.secondary_model = self.train_lowercase()
            self.tag_dataset = self.tag_dataset_rule1

        elif rule == 2:
            """Uses a single model, but the model is a case-insensitive model."""
            self.train = self.train_lowercase
            self.model = self.train()
            self.tag_dataset = self.tag_dataset_baseline

        elif rule == 3:
            """Uses a MLT model, but if the word matches a regex for parsing numbers and fractions, tag as CD"""
            self.train = self.train_baseline
            self.model = self.train()
            self.tag_dataset = self.tag_dataset_numerically_smart

        elif rule == 4:
            self.train = self.train_baseline
            self.model = self.train()
            self.tag_dataset = self.tag_dataset_baseline
            self._DEFAULT_TAG = self.get_most_common_tag()

        elif rule == 5:
            self.train = self.train_baseline
            self.model = self.train()
            self.tag_dataset = self.tag_dataset_rule5

        elif rule is None:
            self.train: Callable = self.train_baseline
            self.tag_dataset: Callable = self.tag_dataset_baseline
            self.model = self.train()

        else:
            raise ValueError("Invalid value for `rule`. If given, must be one of [1, 2, 3, 4, 5]")

    def get_most_common_tag(self) -> str:
        """
        Returns the most common tag in the tagged dataset
        :return: The tag most common in the daatset
        :rtype: str
        """
        counter = defaultdict(int)
        for word in self.dataset:
            counter[word.tag] += 1
        return max(counter, key=counter.get)

    def train_baseline(self) -> dict[str: str]:
        """
        Implements a Most Likely Tag tagger. Given a word, counts all of the tag occurrences for the word and creates
        a model to assign the highest likelihood tag.
        :return:
        :rtype:
        """
        occurrences_counter = defaultdict(lambda: defaultdict(int))
        final_counter = dict()
        for tagged_word in self.dataset:
            occurrences_counter[tagged_word.word][tagged_word.tag] += 1
        for word, tags in occurrences_counter.items():
            final_counter[word] = max(tags, key=tags.get)
        return final_counter

    def train_lowercase(self) -> dict[str: str]:
        """
        Implements a Most Likely Tag tagger but ignores case.
        :return: model as a dict of key=word value=tag
        :rtype: dict
        """
        occurrences_counter = defaultdict(lambda: defaultdict(int))
        final_counter = dict()
        for tagged_word in self.dataset:
            occurrences_counter[tagged_word.word.lower()][tagged_word.tag] += 1
        for word, tags in occurrences_counter.items():
            final_counter[word] = max(tags, key=tags.get)
        return final_counter

    def predict_tag(self, word: str) -> str:
        return self.model.get(word, self._DEFAULT_TAG)

    def predict_tag_w_secondary_model(self, word: str) -> str:
        """
        If the word is not present in the first model, check the secondary models
        :param word: The word to tag
        :return str: The tag for the word
        """
        tag = self.baseline_model.get(word, None)  # Get the tag if word present in dict, else get None
        if not tag:
            tag = self.secondary_model.get(word, self._DEFAULT_TAG)
        return tag

    def tag_dataset_baseline(self, dataset_path: Path) -> str:
        """
        The baseline tagger. Simply chooses the tag from a single model.
        :param dataset_path: The dataset to tag
        :type dataset_path: Path
        :return: A tagged dataset as a string, ready to write to a file
        :rtype: string
        """
        dataset = dataset_path.read_text().replace('[', '').replace(']', '').split()
        tagged_dataset = []
        for word in dataset:
            tag_word = TaggedWord(tag=self.predict_tag(word), word=word)
            tagged_dataset.append(tag_word)
        return ' '.join([str(x) for x in tagged_dataset])

    def tag_dataset_rule1(self, dataset_path: Path) -> str:
        """
        Baseline model, but if the tag is not present in the model, then checks if the lowercase word is present
        in the lowercase model.
        :param dataset_path:
        :type dataset_path:
        :return str: String of tagged words
        """
        dataset = dataset_path.read_text().replace('[', '').replace(']', '').split()
        tagged_dataset = []
        for word in dataset:
            tag_word = TaggedWord(tag=self.predict_tag_w_secondary_model(word), word=word)
            tagged_dataset.append(tag_word)
        return ' '.join([str(x) for x in tagged_dataset])

    def tag_dataset_numerically_smart(self, dataset_path: Path) -> str:
        """
        Baseline model, but if the tag is not present in the model, then checks if the lowercase word is present
        in the lowercase model.
        :param dataset_path:
        :type dataset_path:
        :return str: String of tagged words
        """
        dataset = dataset_path.read_text().replace('[', '').replace(']', '').split()
        tagged_dataset = []
        numerical_pattern = "(\d+|\d{1,3}(,\d{3})*)(\.\d+)?"
        numerical_pattern = f"{numerical_pattern}(/{numerical_pattern})?"
        pattern = re.compile(numerical_pattern)
        for word in dataset:
            if re.fullmatch(pattern, word):
                tag_word = TaggedWord(tag="CD", word=word)
            else:
                tag_word = TaggedWord(tag=self.predict_tag(word), word=word)
            tagged_dataset.append(tag_word)
        return ' '.join([str(x) for x in tagged_dataset])

    def tag_dataset_rule5(self, dataset_path: Path) -> str:
        """
        Baseline model, but if the tag is in a list of commonly confused tags, will choose the tag the predicted tag is confused with
        """
        dataset = dataset_path.read_text().replace('[', '').replace(']', '').split()
        tagged_dataset = []
        substitutions = {
            "FW": "NN",
            "RBS": "JJS"
        }
        for word in dataset:
            predicted_tag = self.predict_tag(word)

            # If the predicted tag is in substitutions, choose the substituted tag. Otherwise, use the predicted tag.
            predicted_tag = substitutions.get(predicted_tag, predicted_tag)
            tag_word = TaggedWord(tag=predicted_tag, word=word)
            tagged_dataset.append(tag_word)
        return ' '.join([str(x) for x in tagged_dataset])


def main():
    parser = argparse.ArgumentParser(description="Train a POS Tagger on a corpus and test on test data")
    parser.add_argument(dest='training', type=Path)
    parser.add_argument(dest='test', type=Path)
    parser.add_argument('-r', dest='rule', required=False, type=int)
    args = parser.parse_args()

    tagger = Tagger(dataset_path=args.training, rule=args.rule)
    newly_tagged = tagger.tag_dataset(args.test)
    print(newly_tagged)


if __name__ == '__main__':
    main()
