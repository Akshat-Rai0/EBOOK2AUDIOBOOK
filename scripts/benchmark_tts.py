#!/usr/bin/env python3
"""
Benchmark TTS engines (VITS and XTTS).

Measures:
- Cold-load time (first model load)
- Synthesis time for a fixed sentence
- RTF (real-time factor = synthesis time / audio duration)
- Peak RSS memory usage

Usage:
    uv run python scripts/benchmark_tts.py
"""

import gc
import os
import resource
import time
import tracemalloc
from pathlib import Path

from ebook2audiobook.models.cast import VoiceRef
from ebook2audiobook.tts.factory import get_engine

# Fixed test sentence (English, ~75 characters)
TEST_SENTENCE = "The quick brown fox jumps over the lazy dog and runs away."


def get_peak_rss_mb() -> float:
    """Get peak RSS memory usage in MB (macOS/Linux)."""
    if os.name == "nt":
        # Windows: use psutil if available
        try:
            import psutil

            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    else:
        # Unix: use resource module
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def benchmark_engine(engine_name: str, device: str = "cpu") -> dict:
    """Benchmark a single TTS engine."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {engine_name} (device: {device})")
    print(f"{'=' * 60}")

    results = {
        "engine": engine_name,
        "device": device,
        "cold_load_time_s": 0.0,
        "synthesis_time_s": 0.0,
        "rtf": 0.0,
        "peak_rss_mb": 0.0,
        "error": None,
    }

    # Clear any cached state
    gc.collect()

    # Measure cold load time
    print(f"Loading {engine_name} model...")
    tracemalloc.start()
    load_start = time.time()

    try:
        engine = get_engine(engine_name, device=device)
        load_time = time.time() - load_start
        results["cold_load_time_s"] = load_time
        print(f"  Cold load time: {load_time:.2f}s")

        # Get peak memory after load
        current_rss = get_peak_rss_mb()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results["peak_rss_mb"] = current_rss
        print(f"  Peak RSS: {current_rss:.1f} MB")
        print(f"  Peak memory (tracemalloc): {peak_mem / 1024 / 1024:.1f} MB")

        # List available voices
        voices = engine.list_voices()
        print(f"  Available voices: {len(voices)}")
        if voices:
            print(f"  First voice: {voices[0]}")

        # Synthesize test sentence
        if voices:
            voice_id = voices[0]["id"]
        else:
            voice_id = "default"
        voice_ref = VoiceRef(engine=engine_name, voice_id=voice_id)
        print(f"\nSynthesizing test sentence with voice: {voice_id}")
        print(f"  Text: {TEST_SENTENCE}")

        synth_start = time.time()
        wav_bytes = engine.synthesize(TEST_SENTENCE, voice_ref)
        synth_time = time.time() - synth_start
        results["synthesis_time_s"] = synth_time
        print(f"  Synthesis time: {synth_time:.2f}s")

        # Calculate audio duration from WAV header
        # WAV format: 44 bytes header, then PCM data
        # Duration = (data_size) / (sample_rate * channels * bytes_per_sample)
        if len(wav_bytes) > 44:
            # Get actual sample rate from engine
            sample_rate = engine.sample_rate
            data_size = len(wav_bytes) - 44
            duration = data_size / (sample_rate * 2)  # 2 bytes per sample (16-bit)
            rtf = synth_time / duration if duration > 0 else 0
            results["rtf"] = rtf
            print(f"  Audio duration: {duration:.2f}s")
            print(f"  RTF (real-time factor): {rtf:.2f}x")
            print("    (RTF < 1.0 = faster than real-time, > 1.0 = slower)")

    except Exception as e:
        results["error"] = str(e)
        print(f"  ERROR: {e}")
        tracemalloc.stop()

    return results


def print_summary(results_list: list[dict]) -> None:
    """Print benchmark summary table."""
    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    header = (
        f"{'Engine':<10} {'Device':<6} {'Load (s)':<10} "
        f"{'Synth (s)':<10} {'RTF':<8} {'RSS (MB)':<10}"
    )
    print(header)
    print("-" * 80)

    for r in results_list:
        if r["error"]:
            print(f"{r['engine']:<10} {r['device']:<6} {'ERROR':<10} {'':<10} {'':<8} {'':<10}")
            print(f"           Error: {r['error']}")
        else:
            print(
                f"{r['engine']:<10} {r['device']:<6} "
                f"{r['cold_load_time_s']:<10.2f} "
                f"{r['synthesis_time_s']:<10.2f} "
                f"{r['rtf']:<8.2f} "
                f"{r['peak_rss_mb']:<10.1f}"
            )

    print("-" * 80)


def main() -> None:
    """Run benchmarks for all available engines."""
    print("TTS Engine Benchmark")
    print(f"Test sentence: {TEST_SENTENCE}")
    print(f"Platform: {os.uname().sysname} {os.uname().machine}")

    results = []

    # Benchmark VITS on CPU
    try:
        results.append(benchmark_engine("vits", device="cpu"))
    except Exception as e:
        print(f"VITS benchmark failed: {e}")
        results.append({"engine": "vits", "device": "cpu", "error": str(e)})

    # Benchmark XTTS on CPU
    try:
        results.append(benchmark_engine("xtts", device="cpu"))
    except Exception as e:
        print(f"XTTS CPU benchmark failed: {e}")
        results.append({"engine": "xtts", "device": "cpu", "error": str(e)})

    # Benchmark XTTS on MPS (macOS only)
    if os.uname().sysname == "Darwin":
        try:
            results.append(benchmark_engine("xtts", device="mps"))
        except Exception as e:
            print(f"XTTS MPS benchmark failed: {e}")
            results.append({"engine": "xtts", "device": "mps", "error": str(e)})

    print_summary(results)

    # Save results to markdown file
    save_to_markdown(results)


def save_to_markdown(results: list[dict]) -> None:
    """Save benchmark results to a markdown file."""
    machine_slug = f"{os.uname().sysname}-{os.uname().machine}".lower()
    output_path = Path("docs/benchmarks") / f"{machine_slug}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"# TTS Benchmark: {os.uname().sysname} {os.uname().machine}\n\n")
        f.write(f"Test sentence: {TEST_SENTENCE}\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Engine | Device | Load (s) | Synth (s) | RTF | RSS (MB) |\n")
        f.write("|--------|--------|----------|-----------|-----|----------|\n")

        for r in results:
            if r["error"]:
                f.write(f"| {r['engine']} | {r['device']} | ERROR | - | - | - |\n")
                f.write(f"| *Error: {r['error']}* | | | | | |\n")
            else:
                f.write(
                    f"| {r['engine']} | {r['device']} | "
                    f"{r['cold_load_time_s']:.2f} | "
                    f"{r['synthesis_time_s']:.2f} | "
                    f"{r['rtf']:.2f} | "
                    f"{r['peak_rss_mb']:.1f} |\n"
                )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
