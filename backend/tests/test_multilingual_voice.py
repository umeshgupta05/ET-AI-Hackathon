"""
Test multilingual voice input: generate synthetic audio in multiple Indian
languages, pass through transcribe_and_translate(), and verify that both
original-language text and English translation are returned.

Requires: gtts (pip install gtts), and either a Groq API key or local Whisper.
"""

import asyncio
import json
import sys
import os
import tempfile
from pathlib import Path

# Ensure backend is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Test phrases in 6 languages ──────────────────────────────────────────
# Each tuple: (language_code_for_gtts, bcp47_hint, display_name, text, expected_english_keywords)
TEST_PHRASES = [
    ("hi", "hi", "Hindi",
     "मैं सीबीआई अधिकारी बोल रहा हूँ, आपका आधार कार्ड फ्रॉड में इस्तेमाल हुआ है",
     ["CBI", "officer", "Aadhaar", "fraud"]),
    ("te", "te", "Telugu",
     "మీ బ్యాంక్ ఖాతా బ్లాక్ చేయబడింది, దయచేసి ఈ నంబర్ కు కాల్ చేయండి",
     ["bank", "account", "block", "call"]),
    ("ta", "ta", "Tamil",
     "உங்கள் வங்கிக் கணக்கு முடக்கப்பட்டுள்ளது, உடனடியாக தொடர்பு கொள்ளுங்கள்",
     ["bank", "account", "block", "contact"]),
    ("bn", "bn", "Bengali",
     "আপনার অ্যাকাউন্ট হ্যাক হয়েছে, এখনই পাসওয়ার্ড পরিবর্তন করুন",
     ["account", "hack", "password", "change"]),
    ("mr", "mr", "Marathi",
     "तुमचा फोन नंबर बेकायदेशीर कामासाठी वापरला गेला आहे",
     ["phone", "number", "illegal"]),
    ("gu", "gu", "Gujarati",
     "તમારું બેંક એકાઉન્ટ સસ્પેન્ડ થયું છે, કૃપા કરી OTP આપો",
     ["bank", "account", "suspend", "OTP"]),
]


def generate_audio(text: str, lang: str) -> bytes:
    """Generate speech audio bytes using Google TTS."""
    from gtts import gTTS
    from io import BytesIO

    tts = gTTS(text=text, lang=lang, slow=False)
    buf = BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


async def run_multilingual_test():
    """Run the full multilingual transcription + translation test."""
    from models.speech.transcriber import get_transcriber

    transcriber = get_transcriber()
    await transcriber.initialize()

    print("=" * 72)
    print("MULTILINGUAL VOICE INPUT TEST")
    print("=" * 72)
    print(f"Groq available: {transcriber._groq_available}")
    print(f"Local pipeline: {'loaded' if transcriber._pipeline else 'not loaded'}")
    print()

    results = []

    for gtts_lang, bcp47, name, phrase, expected_keywords in TEST_PHRASES:
        print(f"── {name} ({bcp47}) ──")
        print(f"  Input text: {phrase[:80]}...")

        # Step 1: Generate audio
        try:
            audio_bytes = generate_audio(phrase, gtts_lang)
            print(f"  Audio generated: {len(audio_bytes)} bytes")
        except Exception as e:
            print(f"  ✗ Audio generation failed: {e}")
            results.append({"language": name, "status": "SKIP", "error": str(e)})
            print()
            continue

        # Step 2: Transcribe and translate
        try:
            result = await transcriber.transcribe_and_translate(
                audio_bytes, language=bcp47, use_groq=True
            )

            original = result.get("original_text", "")
            english = result.get("english_text", "")
            was_translated = result.get("translated_to_english", False)
            provider = result.get("provider", "?")
            trans_provider = result.get("translation_provider", "?")

            print(f"  Original text:  {original[:100]}")
            print(f"  English text:   {english[:100]}")
            print(f"  Translated:     {was_translated}")
            print(f"  Provider:       {provider} → {trans_provider}")

            # Step 3: Check if expected English keywords appear
            english_lower = english.lower()
            matched = [kw for kw in expected_keywords if kw.lower() in english_lower]
            missed = [kw for kw in expected_keywords if kw.lower() not in english_lower]

            if matched:
                print(f"  ✓ Keywords found: {matched}")
            if missed:
                print(f"  ⚠ Keywords missed: {missed}")

            status = "PASS" if was_translated and len(matched) >= 1 else "PARTIAL"
            if not english.strip():
                status = "FAIL"

            print(f"  Status: {status}")
            results.append({
                "language": name,
                "status": status,
                "original": original[:80],
                "english": english[:80],
                "translated": was_translated,
                "keywords_matched": matched,
                "keywords_missed": missed,
            })

        except Exception as e:
            print(f"  ✗ Transcription failed: {e}")
            results.append({"language": name, "status": "FAIL", "error": str(e)})

        print()

    # Summary
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        status_icon = "✓" if r["status"] == "PASS" else "⚠" if r["status"] == "PARTIAL" else "✗"
        print(f"  {status_icon} {r['language']:12s} → {r['status']}")
        if "english" in r:
            print(f"    EN: {r['english'][:70]}")

    passed = sum(1 for r in results if r["status"] in ("PASS", "PARTIAL"))
    total = len(results)
    print(f"\n  {passed}/{total} languages processed successfully")
    print("=" * 72)

    return results


if __name__ == "__main__":
    results = asyncio.run(run_multilingual_test())
