"""Basic autosuggest engine for the next word in a sentence."""

from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import product

import numpy as np


class AutoSuggester:
    """Base class for ingesting lists of tokens, storing n-gram statistics, and
    outputting a suggested next token.
    """

    def __init__(self, n_tok_max, weights):
        """Constructor for AutoSuggester class.

        Args:
            n_tok_max (int): the maximum size n-gram for which to store
                statistics
            weights (list): a list of numerical weights n-grams of different
                lengths, starting from 2 and increasing to `n_tok_max`
        """
        self.n_tok_max = n_tok_max
        self.weights = weights

        # `self._ngram_stats` is a list of dictionaries that map n-1 grams to
        # Counters for individual tokens, representing the last token in a
        # n-gram. This list contains dictionaries for n ranging from 2 to
        # self.n_tok_max.
        self._ngram_stats = [defaultdict(Counter) for _ in self.weights]

    def __bool__(self):
        return any(d for d in self.ngram_stats)

    def update(self, tokens):
        """Updates the n-gram statistics with n-grams from the input token
        list.

        Args:
            tokens (list): an ordered list of tokens
        """
        for num_tok in range(2, self.n_tok_max + 1):
            if len(tokens) < num_tok:
                break

            for idx in range(len(tokens) - num_tok + 1):
                ngram = tokens[idx:idx + num_tok]
                self._add_ngram(ngram)

    def get_all_suggestions(self, tokens):
        """Gets all suggestions with weights, for a token to follow the tokens 
        in the input list.

        Args:
            tokens (list): an ordered list of tokens

        Returns:
            suggestions (Counter): a Counter mapping tokens to weights
                describing their likelihood of being the token to follow those
                in `tokens`
        """
        suggestions = Counter()
        for num_tok in range(1, self.n_tok_max):
            if len(tokens) < num_tok:
                break

            n_minus_one_gram = tokens[-num_tok:]
            stats = self.ngram_stats[num_tok - 1]
            weight = self.weights[num_tok - 1]
            key = tuple(n_minus_one_gram)
            if key in stats:
                _suggestions = stats[key]
                suggestions.update(
                    {k: weight * v for k, v in _suggestions.items()})

        return suggestions

    def suggest(self, tokens, n_suggestions):
        """Returns the top `n_suggestions` suggestions for the next token to
        follow the tokens in the input list.

        Args:
            tokens (list): an ordered list of tokens
            n_suggestions (int): number of suggestions to return

        Returns:
            suggestions (list): a list of `n_suggestions` (token, weight) pairs,
                where `token` is suggested to follow `tokens`, and `weight` is
                the relative likelihood for the given suggestion. The list
                represents the highest likelihood suggestions and is ordered
                from highest to lowest weight.
        """
        return self.get_all_suggestions(tokens).most_common(n_suggestions)

    def merge(self, auto_suggester):
        """Merge another AutoSuggester instance into this one.

        Args:
            auto_suggester (AutoSuggester): another AutoSuggester for which the
               counts will be merged into the current one
        """
        for stats_this, stats_other in zip(
                self.ngram_stats, auto_suggester.ngram_stats):
            for k, other_counts in stats_other.items():
                stats_this[k].update(other_counts)

    @property
    def weights(self):
        """Returns the current weights."""
        return self._weights

    @weights.setter
    def weights(self, values):
        if len(values) != self.n_tok_max - 1:
            raise ValueError(
                "len(weights) != n_tok_max - 1. len(weights) = {}, "
                "n_tok_max - 1 = {}".format(len(values), self.n_tok_max - 1)
            )
        self._weights = values

    @property
    def ngram_stats(self):
        return self._ngram_stats

    def _add_ngram(self, ngram):
        if len(ngram) < 2:
            raise ValueError(
                "Can only add n-grams of length 2 or greater, but got {} of "
                "length {}".format(ngram, len(ngram))
            )

        stats = self.ngram_stats[len(ngram) - 2]
        stats[tuple(ngram[:-1])][ngram[-1]] += 1


class AutoSuggesterFitter:
    """Class for fitting the weights of an AutoSuggester using
    cross-validation.
    """

    def __init__(self, n_tok_max, n_weight_values, score_weights=[(5, 1)]):
        """Constructor for AutoSuggesterFitter class.

        Args:
            n_tok_max (int): the maximum size n-gram for which to store
                statistics
            n_weight_values (int): the number of different weight values to
                grid search over for each weight. The total number of grid
                points will be `n_weight_values ** (n_tok_max - 1)`.
            score_weights (list[tuple]): a list of tuples (rank, weight), which
                indicate that if the actual next token is in the top `rank`
                suggestions or lower, then a score proportional to `weight`
                will be given for this token
        """
        self.n_tok_max = n_tok_max
        self.n_weight_values = n_weight_values

        max_weight = max(weight for _, weight in score_weights)
        # normalize so that the highest score for any given token is 1
        score_weights = [
            (rank, weight / max_weight) for rank, weight in score_weights]
        self.score_weights = sorted(score_weights, key=lambda t: t[0])

    def fit(self, corpus_folds):
        """Fits an AutoSuggester using cross-validation on the input corpus.

        Args:
            corpus_folds (list[Corpus]): a list of Corpus instances, where each
                instance represents one fold

        Returns:
            an AutoSuggester instance with weights that were fit via
                cross-validation over `corpus_folds`
        """
        num_folds = len(corpus_folds)
        if num_folds < 2:
            raise ValueError(
                "num_folds must be at least 2, but got {}".format(num_folds)
            )

        print("loading folds")
        auto_suggester_combinations, auto_suggester_all = self._get_combined(
            corpus_folds)

        print("getting grid")
        grid = self._get_grid(auto_suggester_all)

        print("Fitting {} grid points".format(len(grid)))
        scores = []
        for weights in grid:
            for auto_suggester in auto_suggester_combinations:
                auto_suggester.weights = weights
            print("weights = {}".format(weights))

            score = self._score_cv(auto_suggester_combinations, corpus_folds)
            scores.append(score)
            print("score = {}".format(score))

        best_weights = grid[np.argmax(scores)]
        auto_suggester_all.weights = best_weights

        return auto_suggester_all

    def score(self, auto_suggester, corpus):
        """Returns a score for the given AutoSuggester's performance on the
        given corpus.

        The score is an average of the per-token score over the corpus. The
        per-token score depends on how the actual token compares to the
        suggestion list for that token. In particular, the score is assigned
        using `self.score_weights`.

        Args:
            auto_suggester (AutoSuggester): the AutoSuggester being scored
            corpus (Corpus): the Corpus on which to score `auto_suggester`

        Returns:
            a numerical score for the performance of `auto_suggester` on
                `corpus`
        """
        sum_score = 0
        num_tokens = 0
        for tokens in corpus:
            for idx_end in range(2, len(tokens) + 1):
                sub_tokens = tokens[:idx_end]
                sum_score += self._score_last_token(auto_suggester, sub_tokens)
                num_tokens += 1

        return sum_score / num_tokens

    def _get_combined(self, corpus_folds):
        # `auto_suggester_folds` is a list of AutoSuggester instances, each
        # trained on one fold in `corpus_folds`
        auto_suggester_folds = [
            AutoSuggester(
                self.n_tok_max, [1 for _ in range(self.n_tok_max - 1)])
            for _ in corpus_folds
        ]
        auto_suggester_all = AutoSuggester(
            self.n_tok_max, [1 for _ in range(self.n_tok_max - 1)])
        for auto_suggester, corpus in zip(auto_suggester_folds, corpus_folds):
            for tokens in corpus:
                auto_suggester.update(tokens)
            auto_suggester_all.merge(auto_suggester)

        # `auto_suggester_combinations` is a list of AutoSuggester instances,
        # each one trained on all folds EXCEPT the one with the same index in
        # the list `auto_suggester_folds`
        auto_suggester_combinations = [
            AutoSuggester(
                self.n_tok_max, [1 for _ in range(self.n_tok_max - 1)])
            for _ in corpus_folds
        ]
        for idx, auto_suggester in enumerate(auto_suggester_combinations):
            for idx_fold, auto_suggester_fold in enumerate(
                    auto_suggester_folds):
                if idx != idx_fold:
                    auto_suggester.merge(auto_suggester_fold)

        return auto_suggester_combinations, auto_suggester_all

    def _score_cv(self, auto_suggester_combinations, corpus_folds):
        scores = []
        for auto_suggester, corpus in zip(
                auto_suggester_combinations, corpus_folds):
            scores.append(self.score(auto_suggester, corpus))
        return np.mean(scores)

    def _score_last_token(self, auto_suggester, tokens):
        n_suggestions = self.score_weights[-1][0]
        suggestions = auto_suggester.suggest(tokens[:-1], n_suggestions)
        last_token = tokens[-1]
        ranks = [
            idx + 1 for idx, (token, _) in enumerate(suggestions)
            if token == last_token
        ]

        if not ranks:
            return 0

        rank = ranks[0]
        pos_rank = bisect_left(self.score_weights, (rank, -1))
        if pos_rank == len(self.score_weights):
            return 0
        return self.score_weights[pos_rank][1]

    def _get_grid(self, auto_suggester):
        # bi-grams always have a weight of 1 so that equivalent weights
        # with different normalizations aren't included
        grid_values = [[1]]
        for n_tok in range(3, auto_suggester.n_tok_max + 1):
            print("getting grid values for n-grams of size {}".format(n_tok))
            grid_values.append(
                self._get_weight_percentiles(auto_suggester, n_tok))

        return [list(w) for w in product(*grid_values)]

    def _get_weight_percentiles(self, auto_suggester, n_tokens):
        stats = auto_suggester.ngram_stats[n_tokens - 2]
        bigram_stats = auto_suggester.ngram_stats[0]

        count_ratios = []
        for head_tokens, counts in stats.items():
            first_bigram_token = head_tokens[-1]
            bigram_counts = bigram_stats[(first_bigram_token,)]
            min_bigram_count = min(bigram_counts.values())
            max_bigram_count = max(bigram_counts.values())

            for count in counts.values():
                count_ratios.extend(
                    [min_bigram_count / count, max_bigram_count / count])

        percentiles = np.percentile(
            count_ratios,
            np.arange(0, 100 + 1E-5, 100 / (self.n_weight_values - 1))
        )
        # discard weight values < 1
        return [1] + [w for w in percentiles if w > 1]


class Corpus:
    """A class representing a corpus or document, that can iterate over
    tokenized sentences.
    """

    def __iter__(self):
        """Iterates over lists of tokens, where each list represents a
        tokenized sentence in the corpus.
        """
        raise NotImplementedError("subclasses must implement __iter__()")
