"""
Unpack the labelled TRAIN folders of other RDD2022 countries (Japan, Czech, Norway, United_States, ...)
next to India, from the same figshare release that scripts/fetch_rdd2022_india.py uses.

Why: the held-out India test set holds only 8 transverse-crack (D10) boxes, far too few to say anything about
that class. Other countries' labelled photographs add D10 training examples. They are a different domain
(different roads, cameras, markings), so the results must be reported per source and never presented as India
performance.

    python -m scripts.fetch_rdd2022_countries Japan Czech Norway --local-zip /content/RDD2022_all.zip

Prints one "<Country> <path to its train folder>" line per country. Exits non-zero if a country is not found
(no silent skipping). Only the country's train/ folder is extracted; the official test/ photographs carry no
public labels and are not used.
"""
import argparse
import os
import shutil
import sys
import zipfile

from scripts import fetch_rdd2022_india as _india

OUT = _india.OUT
KNOWN = ("Czech", "India", "Japan", "Norway", "United_States", "China_Drone", "China_MotorBike")


def country_root(country, out=None):
    """Folder named <country> that contains train/, anywhere under out; None if absent."""
    out = out or OUT
    for base, dirs, _files in os.walk(out):
        if os.path.basename(base).lower() == country.lower() and "train" in dirs:
            return base
    return None


def _members(names, country):
    """Archive members under <country>/train/ (case-insensitive), at any depth."""
    out = []
    for n in names:
        parts = n.lower().split("/")
        c = country.lower()
        if c in parts and "train" in parts[parts.index(c) + 1:]:
            out.append(n)
    return out


def extract_country(zpath, country, out=None):
    """Extract <country>/train from a per-country zip, a full release, or a country zip nested inside it."""
    out = out or OUT
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        nested = [n for n in names if n.lower().endswith(".zip") and country.lower() in n.lower()]
        if nested:
            inner = os.path.join(out, os.path.basename(nested[0]))
            with z.open(nested[0]) as src, open(inner, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
            with zipfile.ZipFile(inner) as zi:
                zi.extractall(out)
            os.remove(inner)
            return
        members = _members(names, country)
        if not members:
            raise SystemExit(f"no labelled {country}/train data inside {zpath}")
        z.extractall(out, members=members)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("countries", nargs="+")
    ap.add_argument("--local-zip", default=None, help="the full RDD2022 archive, already downloaded")
    a = ap.parse_args(argv)
    bad = [c for c in a.countries if c not in KNOWN]
    if bad:
        sys.exit(f"unknown country {bad}; known: {', '.join(KNOWN)}")
    todo = [c for c in a.countries if not country_root(c)]
    if todo:
        zpath = a.local_zip
        if not zpath or not zipfile.is_zipfile(zpath):
            os.makedirs(OUT, exist_ok=True)
            zpath = _india.download(_india.FIGSHARE_ALL, os.path.join(OUT, "RDD2022_all.zip"))
        if not zpath:
            sys.exit("the RDD2022 archive could not be downloaded; upload it and pass --local-zip")
        for c in todo:
            extract_country(zpath, c)
    roots = {}
    for c in a.countries:
        r = country_root(c)
        if not r:
            sys.exit(f"{c}: archive extracted but no {c}/train folder was found")
        roots[c] = r
        print(c, os.path.join(r, "train"))
    return roots


if __name__ == "__main__":
    main()
