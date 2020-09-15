import re


class Hypothesis(object):
    def __init__(self, tokens, log_probs, hidden_state, cell_state, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.hidden_state = hidden_state
        self.cell_state = cell_state
        self.coverage = coverage

    def extend(self, token, log_prob, hidden_state, cell_state, coverage):
        return Hypothesis(tokens=self.tokens + [token],
                          log_probs=self.log_probs + [log_prob],
                          hidden_state=hidden_state,
                          cell_state=cell_state,
                          coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


def postprocess(tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True):
    if skip_special_tokens:
        tokens = [t for t in tokens if not is_special(t)]
    out_string = ' '.join(tokens)
    if clean_up_tokenization_spaces:
        out_string = clean_up_tokenization(out_string)
    return out_string


def is_special(token):
    res = re.search("\[[A-Z]+\]", token)
    if res is None:
        return False
    return token == res.group()


def clean_up_tokenization(out_string):
    """
    Reference : transformers.tokenization_utils_base
    Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

    Args:
        out_string (:obj:`str`): The text to clean up.

    Returns:
        :obj:`str`: The cleaned-up string.
    """
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string