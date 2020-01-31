"""Basic autosuggest engine for the next word in a sentence."""

from collections import Counter, defaultdict


class AutoCompleter:
    """Base class for ingesting lists of tokens, storing n-gram statistics, and
    outputting a suggested next token.
    """

    def __init__(self, n_tok_max, weights):
        """Constructor for AutoCompleter class.

        Args:
            n_tok_max (int): the maximum size n-gram for which to store
                statistics
            weights (list): a list of numerical weights n-grams of different
                lengths, starting from 2 and increasing to `n_tok_max`
        """
        self.n_tok_max = n_tok_max
        if len(weights) != n_tok_max - 1:
            raise ValueError(
                "len(weights) != n_tok_max. len(weights) = {}, "
                "n_tok_max = {}".format(len(weight), n_tok_max)
            )
        self.weights = weights

        # `self._ngram_stats` is a list of dictionaries that map n-1 grams to
        # Counters for individual tokens, representing the last token in a
        # n-gram. This list contains dictionaries for n ranging from 2 to
        # self.n_tok_max.
        self._ngram_stats = [defaultdict(Counter) for _ in self.weights]

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
            stats = self._ngram_stats[num_tok - 1]
            weight = self.weights[num_tok - 1]
            _suggestions = stats[tuple(n_minus_one_gram)]
            suggestions += Counter(
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

    def _add_ngram(self, ngram):
        if len(ngram) < 2:
            raise ValueError(
                "Can only add n-grams of length 2 or greater, but got {} of "
                "length {}".format(ngram, len(ngram))
            )

        stats = self._ngram_stats[len(ngram) - 2]
        stats[tuple(ngram[:-1])][ngram[-1]] += 1
