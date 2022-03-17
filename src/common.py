import re
from dataclasses import dataclass

TAG_DELIMITER = re.compile(r"(?<!\\)/")


@dataclass
class TaggedWord:
    word: str
    tag: str

    def __post_init__(self):
        self.tag = self.tag.split('|')[0]  # If two possible tags present, use the first tag

    def __str__(self):
        return f"{self.word}/{self.tag}"


def load_tagged_dataset(dataset_path) -> list[TaggedWord]:
    contents = dataset_path.read_text().replace('[', '').replace(']', '').split()
    tagged_words = []
    for word in contents:
        word, tag = re.split(TAG_DELIMITER, word)
        tagged_words.append(TaggedWord(word=word, tag=tag))
    return tagged_words
