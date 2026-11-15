"""Offline tests for the vision check's parsing and messages. No images downloaded, no Nemotron calls.

Run in the workspace terminal:  python -m fieldvision_agents.vision.test_vision
"""

from .vision import build_messages, parse_reply


def test_parse_clean_json():
    v = parse_reply('{"before_canopy": "green", "after_ground": "bare_soil", "confidence": 0.8, "reason": "dark soil"}')
    assert v["after_ground"] == "bare_soil" and v["hint"] == "silage-like" and v["confidence"] == 0.8


def test_parse_json_inside_text_and_thinking():
    v = parse_reply('<think>looking</think>Sure! {"before_canopy": "brown", "after_ground": "residue", '
                    '"confidence": 0.6, "reason": "stalks"} hope that helps')
    assert v["after_ground"] == "residue" and v["hint"] == "grain-like"


def test_parse_rejects_bad_values():
    v = parse_reply('{"before_canopy": "purple", "after_ground": "lava", "confidence": 7}')
    assert v["before_canopy"] == "unclear" and v["after_ground"] == "unclear" and v["confidence"] == 1.0
    assert parse_reply("no json here")["hint"] == "no hint"


def test_messages_carry_two_images():
    msgs = build_messages(b"\x89PNG-before", b"\x89PNG-after")
    parts = msgs[0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
