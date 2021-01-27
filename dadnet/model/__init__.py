from dadnet.model.feed_forward import FeedForward, LogisticRegression, SimpleFeedForward

ALIASES = dict(
    simpleff=SimpleFeedForward,
    feedforward=FeedForward,
    linear=FeedForward,
    logisticregression=LogisticRegression,
)


def get_model(name):
    return ALIASES[name]
