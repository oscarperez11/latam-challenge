import os
import pytest


@pytest.fixture(autouse=True)
def change_test_dir(monkeypatch):
    challenge_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../challenge")
    monkeypatch.chdir(challenge_dir)
