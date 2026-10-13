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


def _score(**counts: int) -> quality.QualityScore:
    return quality.QualityScore(
        **({"words": 5000, "loop_words": 0, "credit_hits": 0, "hotword_pairs": 0} | counts)
    )


def test_flags_is_empty_for_a_clean_score():
    assert quality.flags(_score()) == ()


def test_flags_trips_loops_at_forty_loop_words_and_not_below():
    assert quality.flags(_score(loop_words=39)) == ()
    assert quality.flags(_score(loop_words=40)) == ("loops",)


def test_flags_trips_credits_on_a_single_credit_hit():
    assert quality.flags(_score(credit_hits=1)) == ("credits",)


def test_flags_trips_hotword_leak_on_a_single_adjacent_pair():
    assert quality.flags(_score(hotword_pairs=1)) == ("hotword_leak",)


def test_flags_lists_every_tripped_flag_in_a_fixed_order():
    assert quality.flags(_score(hotword_pairs=3, credit_hits=2, loop_words=120)) == (
        "loops",
        "credits",
        "hotword_leak",
    )


def test_clean_words_of_clean_prose_is_its_word_count():
    assert quality.clean_words(_PROSE, hotwords=_HOTWORDS) == quality.score(_PROSE, hotwords=_HOTWORDS).words


def test_clean_words_leaves_out_loop_repeats_credit_lines_and_hotword_pairs():
    # ADR-0022 rule 6: the re-transcription gate compares speech, not the junk around it.
    looped = _PROSE + " " + "We lead with mercy. " * 10
    credited = _PROSE + " Subtitles by the Amara.org community"
    leaked = _PROSE + " Ready To Serve, Grace Presbyterian"
    base = quality.clean_words(_PROSE, hotwords=_HOTWORDS)

    assert quality.clean_words(looped, hotwords=None) == base + 4
    assert quality.clean_words(credited, hotwords=None) == base + 2  # "the community" survives
    assert quality.clean_words(leaked, hotwords=_HOTWORDS) == base


def test_clean_words_counts_hotword_terms_as_speech_when_none_were_sent():
    leaked = _PROSE + " Ready To Serve, Grace Presbyterian"
    assert quality.clean_words(leaked, hotwords=None) == quality.clean_words(_PROSE, hotwords=None) + 5
