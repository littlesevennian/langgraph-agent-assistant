import json
import os


PROFILE_PATH = "user_profile.json"


def load_profile():

    if not os.path.exists(PROFILE_PATH):
        return {}

    with open(
        PROFILE_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def save_profile(profile):

    with open(
        PROFILE_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            profile,
            f,
            ensure_ascii=False,
            indent=4
        )