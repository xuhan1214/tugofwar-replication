"""
_validate.py
============
Tiny helper module shared by all v_*.py scripts.
Provides a uniform "PASS / FAIL" reporter so the user gets the same
look and feel across every validation step.

Usage in a v_*.py file:
    from _validate import Checker
    chk = Checker("Phase 2 — Fama-French factors")

    chk.between("monthly RF mean (% per month)", rf_mean*100, 0.10, 0.40)
    chk.equal("FF daily columns", set(ffd.columns), {"date","mktrf","smb","hml","rf","umd"})
    chk.note("monthly mkt-rf mean = %.3f%%" % (mkt_mean*100))

    chk.summary()      # prints PASS / FAIL totals, exits non-zero on FAIL
"""
from __future__ import annotations
import sys


GREEN = "\033[92m"
RED   = "\033[91m"
YEL   = "\033[93m"
DIM   = "\033[2m"
END   = "\033[0m"


class Checker:
    def __init__(self, title: str):
        self.title = title
        self.passed = 0
        self.failed = 0
        self.notes  = 0
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    # ------------- core helpers -------------
    def _ok(self, msg):
        print(f"  {GREEN}✓ PASS{END}  {msg}")
        self.passed += 1

    def _no(self, msg, detail=""):
        print(f"  {RED}✗ FAIL{END}  {msg}")
        if detail:
            print(f"          {DIM}{detail}{END}")
        self.failed += 1

    # ------------- public assertions -------------
    def between(self, name, value, lo, hi):
        """value should fall in [lo, hi]."""
        if value is None or (isinstance(value, float) and (value != value)):
            self._no(f"{name} is missing/NaN")
            return
        if lo <= value <= hi:
            self._ok(f"{name} = {value:.4g}  (expected {lo}..{hi})")
        else:
            self._no(f"{name} = {value:.4g}  (expected {lo}..{hi})")

    def at_least(self, name, value, lo):
        if value is None:
            self._no(f"{name} is missing")
            return
        if value >= lo:
            self._ok(f"{name} = {value:,}  (expected >= {lo:,})")
        else:
            self._no(f"{name} = {value:,}  (expected >= {lo:,})")

    def at_most(self, name, value, hi):
        if value is None:
            self._no(f"{name} is missing")
            return
        if value <= hi:
            self._ok(f"{name} = {value:,}  (expected <= {hi:,})")
        else:
            self._no(f"{name} = {value:,}  (expected <= {hi:,})")

    def equal(self, name, got, expected):
        if got == expected:
            self._ok(f"{name} matches")
        else:
            self._no(f"{name} mismatch", f"got={got!r}\n          expected={expected!r}")

    def is_true(self, name, condition, detail=""):
        if condition:
            self._ok(name)
        else:
            self._no(name, detail)

    # informational
    def note(self, msg):
        print(f"  {YEL}·{END} {msg}")
        self.notes += 1

    def section(self, title):
        print(f"\n  {DIM}--- {title} ---{END}")

    def require_files(self, paths, hint=""):
        """Hard-stop if any of these files don't exist."""
        from pathlib import Path
        missing = [str(p) for p in paths if not Path(p).exists()]
        if missing:
            print(f"  {RED}✗ FAIL{END}  Missing prerequisite file(s):")
            for m in missing:
                print(f"          {DIM}{m}{END}")
            if hint:
                print(f"  {YEL}  Hint: {hint}{END}")
            sys.exit(1)

    # ------------- finalize -------------
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'-'*60}")
        if self.failed == 0:
            print(f"{GREEN}  ALL {self.passed}/{total} CHECKS PASSED{END}")
            print(f"{'-'*60}\n")
            return 0
        else:
            print(f"{RED}  {self.failed}/{total} CHECKS FAILED{END}  ({self.passed} passed)")
            print(f"{'-'*60}\n")
            print(f"{YEL}  Don't proceed to the next phase until failures are resolved.{END}\n")
            sys.exit(1)
