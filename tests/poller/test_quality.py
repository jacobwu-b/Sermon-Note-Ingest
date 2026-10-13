from poller import quality

# Synthetic sermon-like text only; real transcripts never enter this repo (ADR-0004).
_PROSE = (
    "This morning we are continuing in the book of Romans, chapter eight. Paul has spent seven "
    "chapters telling us what God has done, and now he turns to what that means for the way we live. "
    "If you have your Bible, turn there with me, and let's read it together before we pray."
)
_HOTWORDS = "Ready To Serve, Grace Presbyterian, David Jones, Iron Kim, Rod Miles"


def test_score_of_clean_prose_counts_its_words_and_flags_nothing():
    score = quality.score(_PROSE, hotwords=_HOTWORDS)
    assert score == quality.QualityScore(words=53, loop_words=0, credit_hits=0, hotword_pairs=0)


def test_loop_words_counts_only_the_repeats_beyond_the_first_occurrence():
    # "We lead with mercy." x10: nine 4-word repeats were added by the loop.
    text = _PROSE + " " + "We lead with mercy. " * 10 + "And that is the gospel."
    assert quality.loop_words(text) == 36


def test_loop_words_counts_a_long_sentence_loop_once_even_though_its_doubles_also_repeat():
    sentence = "I grew up in a small town, and I'm there in Kentucky. "
    assert quality.loop_words(_PROSE + " " + sentence * 8) == 7 * 12


def test_loop_words_ignores_case_and_punctuation_between_repeats():
    text = "Keep your hand up high! keep your HAND up high, Keep your hand up high."
    assert quality.loop_words(text) == 10


def test_loop_words_ignores_a_phrase_said_only_twice():
    text = "And then I saw a new heaven. And then I saw a new heaven. And the sea was no more."
    assert quality.loop_words(text) == 0


def test_loop_words_ignores_short_liturgical_responses():
    # Under four words a repeat is the congregation answering, not the decoder looping.
    assert quality.loop_words("Amen. Amen. Amen. Amen. Amen. He is risen. He is risen. He is risen.") == 0


def test_loop_words_is_zero_for_empty_text():
    assert quality.loop_words("") == 0


def test_credit_hits_finds_whispers_stock_credit_lines():
    text = (
        f"{_PROSE} Subtitles by the Amara.org community. Thanks for watching! "
        "Thank you for watching. Transcription by CastingWords. Captions by GetTranscribed."
    )
    # The Amara line matches twice: "subtitles by" and "amara.org".
    assert quality.credit_hits(text) == 6


def test_credit_hits_is_zero_for_prose_without_credit_strings():
    assert quality.credit_hits(_PROSE) == 0


def test_hotword_pairs_counts_two_adjacent_terms_read_back_as_speech():
    leaked = "Ready To Serve, Grace Presbyterian, David Jones, Iron Kim, Rod Miles. "
    assert quality.hotword_pairs(_PROSE + " " + leaked * 3, _HOTWORDS) == 12


def test_hotword_pairs_ignores_case():
    assert quality.hotword_pairs("grace presbyterian, DAVID JONES spoke", _HOTWORDS) == 1


def test_hotword_pairs_ignores_a_term_spoken_on_its_own():
    text = "Welcome to Grace Presbyterian. David Jones is preaching this morning."
    assert quality.hotword_pairs(text, _HOTWORDS) == 0


def test_hotword_pairs_is_zero_when_no_hotwords_were_sent():
    leaked = "Ready To Serve, Grace Presbyterian, David Jones"
    assert quality.hotword_pairs(leaked, None) == 0
    assert quality.hotword_pairs(leaked, "") == 0


def test_score_combines_every_detector():
    text = (
        _PROSE
        + " We lead with mercy. We lead with mercy. We lead with mercy."
        + " Grace Presbyterian, David Jones. Thanks for watching."
    )
    assert quality.score(text, hotwords=_HOTWORDS) == quality.QualityScore(
        words=53 + 12 + 4 + 3, loop_words=8, credit_hits=1, hotword_pairs=1
    )


def test_score_holds_only_counts_so_it_is_safe_to_log():
    # This repo's Actions logs are public: a score must never carry transcript text.
    text = "A sentinel phrase nobody would say. " * 5
    assert "sentinel" not in repr(quality.score(text, hotwords="sentinel phrase, nobody"))
