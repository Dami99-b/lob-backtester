import numpy as np
from conftest import todo

from lobtest.book import OrderBook
from lobtest.replay_fast import replay_top_of_book
from lobtest.synthetic import generate


@todo
def test_fast_replay_matches_reference():
    msgs = generate(20_000, seed=3)
    fast = replay_top_of_book(msgs)
    book = OrderBook()
    for i, m in enumerate(msgs):
        book.apply(m)
        bid, ask = book.best_bid(), book.best_ask()
        expected = (
            bid[0] if bid else -1,
            bid[1] if bid else -1,
            ask[0] if ask else -1,
            ask[1] if ask else -1,
        )
        assert tuple(int(x) for x in fast[i]) == expected, f"mismatch at message {i}"
    assert np.asarray(fast).shape == (len(msgs), 4)
