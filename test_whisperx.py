"""Quick command-line ASR check kept under the historical filename.

Usage: ``python test_whisperx.py path\\to\\song.mp3 --language en``
The project uses faster-whisper/py-roller, not the old WhisperX prototype.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Проверка распознавания faster-whisper")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--language", default=None)
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Файл не найден: {args.audio}")

    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Модель: {args.model}; устройство: {device}; тип: {compute_type}")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(args.audio), language=args.language, word_timestamps=True)
    print(f"Язык: {info.language} ({info.language_probability:.1%})")
    for segment in segments:
        print(f"[{segment.start:8.3f}–{segment.end:8.3f}] {segment.text.strip()}")
        for word in segment.words or []:
            print(f"  {word.start:8.3f}–{word.end:8.3f}  {word.word.strip()}")


if __name__ == "__main__":
    main()
