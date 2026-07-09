import random
from collections import defaultdict


class BigramModel:
    def __init__(self, corpus):
        self.bigram_counts = defaultdict(list)
        self.train(corpus)

    def train(self, corpus):
        for sentence in corpus:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i + 1]
                self.bigram_counts[current_word].append(next_word)

    def generate_text(self, start_word, length):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_counts.get(current_word)

            if not next_words:
                break

            current_word = random.choice(next_words)
            generated_words.append(current_word)

        return " ".join(generated_words)