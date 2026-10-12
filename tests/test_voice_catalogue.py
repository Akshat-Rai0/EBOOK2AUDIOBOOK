from ebook2audiobook.audio.catalogue import VoiceCatalogue, VoiceEntry
from ebook2audiobook.tts.fake_tts import FakeTTS


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
        # All fake voices have gender="unknown" and gender_verified=False
        # filter_verified should return empty list for gender="male"
        verified_males = cat.filter_verified("fake", gender="male")
        assert len(verified_males) == 0

    def test_unverified_age_bracket_behavior(self):
        cat = VoiceCatalogue()
        # All fake voices have age_bracket="unknown" and age_verified=False
        # filter_verified should return empty list for specific age brackets
        young_adults = cat.filter_verified("fake", age_bracket="young_adult")
        assert len(young_adults) == 0

        elderly = cat.filter_verified("fake", age_bracket="elderly")
        assert len(elderly) == 0

    def test_acoustic_distance_between_voices(self):
        v_deep_male = VoiceEntry(
            id="v1",
            name="Deep",
            engine="fake",
            proxy_median_f0_hz=110.0,
            proxy_spectral_centroid_hz=1400.0,
        )
        v_high_female = VoiceEntry(
            id="v2",
            name="High",
            engine="fake",
            proxy_median_f0_hz=235.0,
            proxy_spectral_centroid_hz=2550.0,
        )
        v_similar_deep = VoiceEntry(
            id="v3",
            name="Deep2",
            engine="fake",
            proxy_median_f0_hz=115.0,
            proxy_spectral_centroid_hz=1450.0,
        )

        dist_far = v_deep_male.acoustic_distance_to(v_high_female)
        dist_near = v_deep_male.acoustic_distance_to(v_similar_deep)

        assert dist_far > 2.0  # Very distinct
        assert dist_near < 0.5  # Very acoustically similar (potential collision)

    def test_acoustic_distance_with_null_proxies(self):
        """Distance should be infinity when either voice lacks proxy measurements."""
        v_with_proxy = VoiceEntry(
            id="v1",
            name="WithProxy",
            engine="fake",
            proxy_median_f0_hz=160.0,
            proxy_spectral_centroid_hz=1900.0,
        )
        v_without_proxy = VoiceEntry(
            id="v2",
            name="WithoutProxy",
            engine="fake",
            proxy_median_f0_hz=None,
            proxy_spectral_centroid_hz=None,
        )

        dist = v_with_proxy.acoustic_distance_to(v_without_proxy)
        assert dist == float("inf")

        dist2 = v_without_proxy.acoustic_distance_to(v_with_proxy)
        assert dist2 == float("inf")

    def test_all_catalogue_ids_exist_in_fake_engine(self):
        """Every catalogue ID should exist in the real engine's voice list."""
        cat = VoiceCatalogue()
        fake_engine = FakeTTS()
        engine_voices = fake_engine.list_voices()
        valid_ids = {v["id"] for v in engine_voices}

        fake_voices = cat.get_entries("fake")
        catalogue_ids = {v.id for v in fake_voices}

        # All catalogue IDs should be in the engine's voice list
        assert catalogue_ids.issubset(valid_ids), (
            f"Catalogue contains IDs not in engine: {catalogue_ids - valid_ids}"
        )
