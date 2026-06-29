"""
sentence_builder.py
--------------------
Turns a stream of per-frame gesture predictions into a sentence.

Problem it solves: your model predicts a label every ~400ms, so a single
held-up "A" would otherwise spam "AAAAAAAAA" into the output. This module
only *commits* a letter once it has been held steadily for N frames, and
adds a cooldown so the same letter can't be committed twice in a row
unless the hand resets (or enough time passes) in between.

Drop this in scripts/ alongside your other pipeline files.
"""

import time


class SentenceBuilder:
    def __init__(
        self,
        stable_frames_required=4,    # how many consecutive matching predictions before commit
        cooldown_seconds=1.0,        # min gap before the same letter can commit twice in a row
        space_label="SPACE",
        delete_label="DELETE",
        no_hand_label="NoHand",
    ):
        self.stable_frames_required = stable_frames_required
        self.cooldown_seconds = cooldown_seconds
        self.space_label = space_label
        self.delete_label = delete_label
        self.no_hand_label = no_hand_label

        self.sentence = ""
        self._current_label = None
        self._stable_count = 0
        self._last_committed_label = None
        self._last_commit_time = 0.0

    def update(self, label):
        """
        Feed in the latest predicted label for this frame/request.
        Returns the committed character/action if one was just locked in
        this call, otherwise None. Call this once per prediction.
        """
        if label is None or label == self.no_hand_label:
            # hand left the frame -> require a fresh hold next time,
            # this is what lets the same letter be signed twice in a row
            self._current_label = None
            self._stable_count = 0
            return None

        if label == self._current_label:
            self._stable_count += 1
        else:
            self._current_label = label
            self._stable_count = 1

        if self._stable_count < self.stable_frames_required:
            return None

        now = time.time()
        same_as_last = label == self._last_committed_label
        cooled_down = (now - self._last_commit_time) >= self.cooldown_seconds

        if same_as_last and not cooled_down:
            return None  # already committed this letter recently, waiting on cooldown

        self._commit(label)
        self._last_committed_label = label
        self._last_commit_time = now
        self._stable_count = 0  # require a fresh hold before this label can commit again
        return label

    def _commit(self, label):
        if label == self.space_label:
            if not self.sentence.endswith(" "):
                self.sentence += " "
        elif label == self.delete_label:
            self.sentence = self.sentence[:-1]
        else:
            self.sentence += label

    def get_sentence(self):
        return self.sentence

    def clear(self):
        self.sentence = ""
        self._current_label = None
        self._stable_count = 0
        self._last_committed_label = None
        self._last_commit_time = 0.0


if __name__ == "__main__":
    # quick manual test - simulates a stream of predictions
    sb = SentenceBuilder(stable_frames_required=3, cooldown_seconds=0.0)
    stream = (
        ["A"] * 3 + ["NoHand"] +
        ["B"] * 3 + ["NoHand"] +
        ["B"] * 3 + ["NoHand"] +   # signing B twice in a row, with a reset between
        ["SPACE"] * 3 + ["NoHand"] +
        ["C"] * 3
    )
    for label in stream:
        committed = sb.update(label)
        if committed:
            print(f"Committed: {committed!r}  ->  sentence so far: {sb.get_sentence()!r}")
    print("Final sentence:", sb.get_sentence())
