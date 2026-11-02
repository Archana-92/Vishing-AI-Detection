"""
generate_dataset.py
--------------------
Generates a SYNTHETIC labeled dataset of call transcripts (text only — no
audio is synthesized by this script; see README for how to pair transcripts
with TTS-generated or recorded audio if real acoustic training is desired).

WHY SYNTHETIC: no public dataset combining audio + transcripts + a 3-way
{scam, legitimate, casual} phone-call label was found (a Kaggle fraud-call
dataset was evaluated and found text-only, with no audio — see README).
This generator produces reproducible, class-balanced template-based
transcripts to (a) validate the end-to-end pipeline runs correctly, and
(b) serve as a documented placeholder/starting point, NOT as a claim of
real-world representativeness. Every row is tagged `synthetic=True` in the
output CSV and this fact must be stated in any paper that uses this data.

Usage:
    python -m vishing_ai.data.generate_dataset --out dataset/ --n-per-class 4000
"""

from __future__ import annotations

import argparse
import csv
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

random.seed(42)

# ---------------------------------------------------------------------
# Template slots, organized by class and subtype for lexical diversity.
# ---------------------------------------------------------------------
SCAM_SUBTYPES: Dict[str, List[str]] = {
    "irs_tax_scam": [
        "This is {agent} calling from the {agency}. Our records show you owe "
        "{amount} in back taxes. If you do not pay {method} within {deadline}, "
        "a warrant will be issued for your arrest. This is your final notice, "
        "please stay on the line and do not hang up.",
        "Hello, I am {agent}, badge number {badge}, contacting you from the "
        "{agency} regarding unpaid taxes of {amount}. You must act now and "
        "settle this {method} today or legal action will begin immediately.",
    ],
    "bank_account_scam": [
        "Hi, this is {agent} from {bank} fraud department. We detected suspicious "
        "activity on your account. Can you please confirm your account number and "
        "verify your password so we can secure your funds right now?",
        "This is an urgent security alert from {bank}. Someone tried to access "
        "your account. Please provide your one time code sent to your phone so "
        "we can verify your identity immediately, don't tell anyone about this call.",
    ],
    "tech_support_scam": [
        "Hello, this is {agent} from {tech_company} technical support. We have "
        "detected a virus on your computer. Please don't hang up, stay on the "
        "line while I guide you to install remote access software right now.",
        "Sir, our systems show your computer sent an error report to us. This is "
        "urgent, you need to purchase a gift card today to renew your protection "
        "plan or your files will be deleted within 24 hours.",
    ],
    "prize_scam": [
        "Congratulations, you have won {amount} in the {sweepstakes} lottery! "
        "To claim your prize today, please wire transfer a small processing fee "
        "using {method}. This offer expires within 24 hours, act now.",
        "This is {agent} calling to confirm you've been selected to win a "
        "{prize}. Just confirm your social security number and pay a small "
        "delivery fee via gift card and it will ship immediately.",
    ],
    "romance_grandparent_scam": [
        "Grandma, it's me, I'm in trouble and I need your help right now. "
        "Please don't tell mom and dad, just send {amount} using {method} "
        "immediately, I'm too embarrassed to explain over the phone right now.",
    ],
}

LEGITIMATE_SUBTYPES: Dict[str, List[str]] = {
    "appointment_reminder": [
        "Hi, this is {agent} calling from {clinic} to remind you about your "
        "appointment on {date} at {time}. Please call us back if you need to "
        "reschedule. Have a great day.",
        "Hello, this is a reminder from {clinic} regarding your upcoming visit "
        "on {date} at {time}. If this time no longer works for you, feel free "
        "to call our office anytime during business hours.",
        "Hi, {agent} here from {clinic}. Just confirming your appointment for "
        "{date} at {time}. Reply or call back if you'd like to reschedule.",
    ],
    "delivery_notification": [
        "Hello, this is {agent} from {company} customer service. Your package "
        "is scheduled to arrive on {date}. Someone needs to be available to "
        "sign for it. Please let us know if that works for you.",
        "Hi, {agent} calling from {company} regarding your recent order. "
        "Delivery is expected around {time} on {date}, please make sure "
        "someone is available.",
    ],
    "customer_service_followup": [
        "Hi, I'm {agent} following up on the support ticket you opened with "
        "{company} on {date}. I wanted to check whether the issue with your "
        "order was resolved to your satisfaction around {time}.",
        "Hello, this is {agent} from {company}. I'm calling to follow up on "
        "your recent inquiry from {date} and see if you have any other "
        "questions for us before {time}.",
        "Hi, {agent} with {company} here, checking back in on the ticket "
        "from {date}. Do you have a few minutes around {time} to talk it "
        "through?",
    ],
    "survey_call": [
        "Good afternoon, this is {agent} conducting a short customer "
        "satisfaction survey for {company}. Would you have two minutes "
        "around {time} to share your feedback about your recent experience?",
        "Hi, {agent} here on behalf of {company}. We're gathering feedback "
        "from customers this {date}, would {time} be a good time for a "
        "quick two minute survey?",
        "Hello, this is {agent} calling for {company}'s customer feedback "
        "line. Do you have a moment on {date} around {time} to answer a "
        "few short questions?",
    ],
}

CASUAL_SUBTYPES: Dict[str, List[str]] = {
    "friend_catchup": [
        "Hey, it's {name}! I was just thinking about you, how have you been? "
        "We should grab {casual_activity} sometime {casual_time} if you're free.",
        "Hi! Long time no talk. How's the family doing? I heard you moved to "
        "{casual_place} recently, I'd love to hear all about it.",
        "Hey {name}, it's been way too long. Are you free {casual_time}? "
        "I'd love to catch up over {casual_activity}.",
        "Hi, it's {name}. I was in {casual_place} the other day and thought "
        "of you. We should catch up over {casual_activity} {casual_time}.",
    ],
    "family_checkin": [
        "Hi honey, it's {name}, just calling to check in and see how your day "
        "is going. Did you eat dinner yet? Call me back {casual_time}.",
        "Hey it's {name}, just checking in on you. How was {casual_topic}? "
        "Let me know if you need anything, love you.",
        "Hi, it's {name} calling from {casual_place}. Just wanted to check "
        "in about {casual_topic}, call me back whenever you're free.",
    ],
    "weekend_plans": [
        "Hey, are we still on for {casual_activity} {casual_time}? Let me "
        "know what time works, I can pick you up if you want.",
        "Hi {name}, want to do {casual_activity} {casual_time}? No pressure "
        "if you're busy, just let me know either way.",
        "Hey {name}, thinking about {casual_activity} {casual_time} near "
        "{casual_place}. Are you in?",
    ],
    "small_talk": [
        "Hi, how's it going? The weather has been crazy lately in {casual_place}, "
        "hasn't it? Anyway, just wanted to say hello, talk soon.",
        "Hey {name}, how's everything at {casual_place}? Just wanted to check "
        "in and say hi, nothing urgent.",
        "Hi {name}, just wanted to chat about {casual_topic} for a bit, "
        "how have things been at {casual_place}?",
    ],
}

SLOTS = {
    "agent": ["John Miller", "Sarah Davis", "Officer Thompson", "Agent Lee", "Michael Brown",
              "Jennifer Clark", "David Wilson", "Amanda Reed"],
    "agency": ["Internal Revenue Service", "Social Security Administration",
               "Federal Trade Commission", "Department of Justice"],
    "amount": ["$1,200", "$3,450", "$780", "$5,000", "$2,300", "$15,000"],
    "method": ["with gift cards", "via wire transfer", "using bitcoin", "with a prepaid card"],
    "deadline": ["24 hours", "48 hours", "one hour", "the next 30 minutes"],
    "badge": ["44213", "80217", "11209", "99871"],
    "bank": ["Chase Bank", "Wells Fargo", "Bank of America", "Citibank"],
    "tech_company": ["Microsoft", "Apple Support", "Norton Security", "Windows Technical Team"],
    "sweepstakes": ["Publishers Clearing House", "National Rewards Center", "Mega Prize Draw"],
    "prize": ["new laptop", "$10,000 cash prize", "free vacation package"],
    "clinic": ["Riverside Medical Center", "Sunrise Family Clinic", "Dr. Patel's office"],
    "company": ["FastShip Logistics", "TechCorp Support", "ValueMart", "GlobalTel"],
    "date": ["Tuesday", "next Monday", "the 14th", "this Friday", "next Wednesday", "the 22nd"],
    "time": ["10:30 AM", "2:00 PM", "9:00 AM", "4:15 PM", "11:45 AM", "6:00 PM"],
    "name": ["Alex", "Jamie", "Sam", "Taylor", "Jordan", "Casey", "Morgan", "Riley",
              "Chris", "Pat", "Drew", "Robin", "Jess", "Charlie", "Sydney", "Dana",
              "Marley", "Reese", "Skyler", "Quinn"],
    "casual_activity": ["coffee", "lunch", "a movie", "the game", "a hike", "dinner",
                         "a walk in the park", "board games", "a bike ride", "brunch",
                         "a picnic", "a museum trip", "karaoke", "a road trip"],
    "casual_time": ["this week", "this weekend", "tomorrow", "next weekend",
                     "on Saturday", "after work", "this afternoon", "sometime soon",
                     "on Sunday", "next Thursday", "later tonight", "this evening"],
    "casual_place": ["downtown", "the new apartment", "the lake house", "Austin",
                      "the old neighborhood", "the new office", "her parents' place",
                      "the beach house", "Chicago", "the new gym", "the farmers market",
                      "the coast"],
    "casual_topic": ["the new job", "the move", "the kids", "the trip next month",
                      "the renovation", "your new car", "the reunion", "the garden",
                      "the game last night", "the new puppy"],
}


def _fill_template(template: str) -> str:
    out = template
    for slot, options in SLOTS.items():
        placeholder = "{" + slot + "}"
        while placeholder in out:
            out = out.replace(placeholder, random.choice(options), 1)
    return out


def _generate_unique(subtypes: Dict[str, List[str]], n: int, label: str) -> List[Tuple[str, str, str]]:
    """Generate n unique (transcript, label, subtype) tuples via retry loop,
    since naive template filling can collide on short templates."""
    seen = set()
    rows = []
    subtype_names = list(subtypes.keys())
    attempts = 0
    max_attempts = n * 100

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        subtype = random.choice(subtype_names)
        template = random.choice(subtypes[subtype])
        text = _fill_template(template)
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, label, subtype))

    if len(rows) < n:
        raise RuntimeError(
            f"Could not generate {n} unique samples for class '{label}' after "
            f"{max_attempts} attempts (got {len(rows)}). Add more template variety."
        )
    return rows


def generate_dataset(n_per_class: int = 4000) -> List[Tuple[str, str, str, bool]]:
    rows = []
    for label, subtypes in [("scam", SCAM_SUBTYPES),
                             ("legitimate", LEGITIMATE_SUBTYPES),
                             ("casual", CASUAL_SUBTYPES)]:
        generated = _generate_unique(subtypes, n_per_class, label)
        rows.extend([(text, label, subtype, True) for text, label, subtype in generated])
    random.shuffle(rows)
    return rows


def write_dataset(rows: List[Tuple[str, str, str, bool]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "transcript", "label", "subtype", "synthetic"])
        for i, (text, label, subtype, synthetic) in enumerate(rows):
            writer.writerow([f"sample_{i:05d}", text, label, subtype, synthetic])

    readme_path = out_dir / "README.md"
    readme_path.write_text(_README_TEXT.format(n=len(rows), n_per_class=len(rows) // 3))

    return csv_path


_README_TEXT = """\
# Synthetic Vishing Dataset

**THIS DATA IS SYNTHETICALLY GENERATED.** It contains {n} labeled call
transcripts ({n_per_class} each for `scam`, `legitimate`, and `casual`),
produced by randomized slot-filling over hand-authored templates
(`generate_dataset.py`). It is NOT a corpus of real phone call recordings
or transcripts.

## Why synthetic
No public dataset was found that pairs real audio + transcripts with a
3-class {{scam, legitimate, casual}} phone-call label. A Kaggle fraud-call
dataset was evaluated and found to be text-only with no audio; its
vocabulary was used only to help seed the scam-keyword list in
`vishing_ai/features/linguistic.py`, not as training data here.

## Intended use & limitations
- Useful for validating that the end-to-end pipeline (preprocessing ->
  feature extraction -> CTFN -> XAI) runs correctly and for illustrating
  the architecture in a research paper.
- **Not suitable, on its own, for claiming real-world generalization
  performance.** Accuracy numbers obtained by training/evaluating solely on
  this dataset reflect performance on synthetic, template-structured text
  and (if paired with TTS audio) synthetic acoustic conditions. Any
  manuscript using this dataset must disclose that fact explicitly and
  should treat such numbers as an upper-bound / pipeline-validation figure,
  not a claim about real vishing-call detection accuracy.
- Templates are hand-authored and finite in number per subtype; the
  generator's `_generate_unique` retry loop guarantees exact per-class
  unique counts but does not guarantee linguistic diversity beyond the
  template + slot-filling space.
- This package ships **transcripts only**. To obtain paired audio, run
  transcripts through a TTS engine of your choice (not included) or record
  human read-throughs; the audio preprocessing pipeline
  (`vishing_ai/preprocessing/audio_preprocessing.py`) expects standard
  audio files and does not care how they were produced.

## Files
- `metadata.csv`: id, transcript, label, subtype, synthetic (always True)
"""


def package_zip(out_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in out_dir.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(out_dir.parent))
    return zip_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="dataset", help="output directory")
    parser.add_argument("--n-per-class", type=int, default=4000)
    parser.add_argument("--zip", action="store_true", help="also package as a .zip")
    args = parser.parse_args()

    out_dir = Path(args.out)
    rows = generate_dataset(n_per_class=args.n_per_class)
    csv_path = write_dataset(rows, out_dir)
    print(f"Wrote {len(rows)} synthetic samples to {csv_path}")

    if args.zip:
        zip_path = out_dir.parent / "synthetic_vishing_dataset.zip"
        package_zip(out_dir, zip_path)
        print(f"Packaged dataset at {zip_path}")


if __name__ == "__main__":
    main()
