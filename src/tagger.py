import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from common import TAG_DELIMITER, TaggedWord, load_tagged_dataset


class Tagger:
    def __init__(self, dataset_path: Path, default_tag: str = 'NN'):
        self.dataset_path = dataset_path
        self.dataset = load_tagged_dataset(self.dataset_path)
        self.model = self.train()
        self._DEFAULT_TAG = default_tag

    def train(self) -> dict[str: str]:
        occurrences_counter = defaultdict(lambda: defaultdict(int))
        final_counter = dict()
        for tagged_word in self.dataset:
            occurrences_counter[tagged_word.word][tagged_word.tag] += 1
        for word, tags in occurrences_counter.items():
            final_counter[word] = max(tags, key=tags.get)
        return final_counter

    def predict_tag(self, word: str) -> str:
        return self.model.get(word, self._DEFAULT_TAG)

    def tag_dataset(self, dataset_path: Path) -> str:
        dataset = dataset_path.read_text().replace('[', '').replace(']', '').split()
        tagged_dataset = []
        for word in dataset:
            tag_word = TaggedWord(tag=self.predict_tag(word), word=word)
            tagged_dataset.append(tag_word)
        return ' '.join([str(x) for x in tagged_dataset])


def main():
    parser = argparse.ArgumentParser(description="Train a POS Tagger on a corpus and test on test data")
    parser.add_argument(dest='training', type=Path)
    parser.add_argument(dest='test', type=Path)
    args = parser.parse_args()
    tagger = Tagger(args.training)
    newly_tagged = tagger.tag_dataset(args.test)
    print(newly_tagged)


if __name__ == '__main__':
    main()
