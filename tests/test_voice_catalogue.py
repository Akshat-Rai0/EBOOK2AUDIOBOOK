from ebook2audiobook.audio.catalogue import VoiceCatalogue, VoiceEntry


class TestVoiceCatalogue:
    def test_catalogue_loads_all_engines(self):
        cat = VoiceCatalogue()
        fake_voices = cat.get_entries("fake")
        vits_voices = cat.get_entries("vits")
        xtts_voices = cat.get_entries("xtts")

        assert len(fake_voices) >= 6
        assert len(vits_voices) >= 100
        assert len(xtts_voices) >= 20

    def test_narrator_reserved_excluded_from_characters(self):
        cat = VoiceCatalogue()
        narrator = cat.get_narrator_voice("fake")
        assert narrator is not None
        assert narrator.id == "fake-narrator"
        assert narrator.narrator_reserved is True

        char_voices = cat.available_for_characters("fake")
        char_ids = {v.id for v in char_voices}
        assert "fake-narrator" not in char_ids
        assert "fake-char-1" in char_ids

    def test_unverified_tags_not_matched_by_filter_verified(self):
        cat = VoiceCatalogue()
        # In fake_voices:
        # fake-char-1: male, verified=True
        # fake-char-3: male, verified=False
        verified_males = cat.filter_verified("fake", gender="male")
        male_ids = [v.id for v in verified_males]

        assert "fake-char-1" in male_ids
        # Unverified voice must NOT be returned by filter_verified
        assert "fake-char-3" not in male_ids

    def test_unverified_age_bracket_behavior(self):
        cat = VoiceCatalogue()
        # fake-char-2: young_adult, age_verified=True
        # fake-char-4: elderly, age_verified=False
        young_adults = cat.filter_verified("fake", age_bracket="young_adult")
        assert any(v.id == "fake-char-2" for v in young_adults)

        elderly = cat.filter_verified("fake", age_bracket="elderly")
        # fake-char-4 is unverified elderly, so should not match
        assert not any(v.id == "fake-char-4" for v in elderly)

    def test_acoustic_distance_between_voices(self):
        v_deep_male = VoiceEntry(
            id="v1",
            name="Deep",
            engine="fake",
            measured_median_f0_hz=110.0,
            measured_spectral_centroid_hz=1400.0,
        )
        v_high_female = VoiceEntry(
            id="v2",
            name="High",
            engine="fake",
            measured_median_f0_hz=235.0,
            measured_spectral_centroid_hz=2550.0,
        )
        v_similar_deep = VoiceEntry(
            id="v3",
            name="Deep2",
            engine="fake",
            measured_median_f0_hz=115.0,
            measured_spectral_centroid_hz=1450.0,
        )

        dist_far = v_deep_male.acoustic_distance_to(v_high_female)
        dist_near = v_deep_male.acoustic_distance_to(v_similar_deep)

        assert dist_far > 2.0  # Very distinct
        assert dist_near < 0.5  # Very acoustically similar (potential collision)
