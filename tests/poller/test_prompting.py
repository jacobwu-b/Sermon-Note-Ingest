from poller import prompting


def _record(**overrides) -> dict:
    record = {
        "guid": "hillside:123",
        "title": "The Cost of Discipleship",
        "series": "Luke",
        "speaker": "Keith Crosby",
        "blurb": "A long paragraph of description that must never be used as a hotword.",
    }
    record.update(overrides)
    return record


def _terms(hotwords: str | None) -> list[str]:
    assert hotwords is not None
    return hotwords.split(", ")


def test_build_hotwords_puts_the_sermons_own_speaker_series_and_title_before_church_terms():
    hotwords = prompting.build_hotwords(
        _record(), church_terms=["Jesse Fenn", "Jono Burlini"], vocabulary=("Hillside Church",)
    )
    assert _terms(hotwords) == [
        "Keith Crosby",
        "Luke",
        "The Cost of Discipleship",
        "Hillside Church",
        "Jesse Fenn",
        "Jono Burlini",
    ]
    assert "long paragraph" not in hotwords


def test_build_hotwords_dedupes_a_speaker_that_also_appears_in_the_church_vocabulary():
    hotwords = prompting.build_hotwords(
        _record(), church_terms=["keith crosby", "Keith Crosby", "Luke"], vocabulary=("Keith Crosby",)
    )
    terms = _terms(hotwords)
    assert terms.count("Keith Crosby") == 1
    assert "keith crosby" not in terms
    assert terms.count("Luke") == 1


def test_build_hotwords_keeps_only_the_first_few_terms_so_the_sermons_own_survive():
    # A long list measurably makes large-v3 skip speech and slow down; the cap keeps
    # the sermon-specific terms and the most frequent church terms, dropping the tail.
    many_names = [f"Guest Preacher {i}" for i in range(50)]
    terms = _terms(prompting.build_hotwords(_record(), church_terms=many_names, vocabulary=()))
    assert len(terms) == prompting.HOTWORDS_MAX_TERMS
    assert terms[:3] == ["Keith Crosby", "Luke", "The Cost of Discipleship"]
    assert terms[3:] == many_names[: prompting.HOTWORDS_MAX_TERMS - 3]


def test_build_hotwords_skips_fields_the_record_does_not_know():
    hotwords = prompting.build_hotwords(
        _record(speaker=None, series="  "), church_terms=["Jesse Fenn"], vocabulary=()
    )
    assert _terms(hotwords) == ["The Cost of Discipleship", "Jesse Fenn"]


def test_build_hotwords_is_none_when_nothing_is_known():
    assert prompting.build_hotwords({"guid": "x"}, church_terms=[], vocabulary=()) is None


def test_church_vocabulary_lists_distinct_speakers_and_series_most_frequent_first_and_skips_empties():
    records = {
        "a": {"speaker": "Jesse Fenn", "series": None},
        "b": {"speaker": "Keith Crosby", "series": "Luke"},
        "c": {"speaker": "Keith Crosby", "series": ""},
        "d": {"speaker": "  ", "series": "Luke"},
        "e": {"speaker": "Keith Crosby"},
    }
    assert prompting.church_vocabulary(records) == ["Keith Crosby", "Luke", "Jesse Fenn"]
