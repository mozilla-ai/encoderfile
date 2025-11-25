#!/usr/bin/env python3

import argparse
from enum import StrEnum
from pathlib import Path
import toml
import re
from pydantic import BaseModel
from typing import Optional, List

SEMVER_RE = re.compile(
    r"^v?(\d+)\.(\d+)\.(\d+)"
    r"(?:-(alpha|beta|rc)\.(\d+))?"
    r"(?:\+([\w\.]+))?$"
)

class Pre(StrEnum):
    ALPHA = "alpha"
    BETA = "beta"
    RC = "rc"
    RELEASE = "release"

class Version(BaseModel):
    mjr: int
    mnr: int
    patch: int
    pre: Pre = Pre.RELEASE
    n: Optional[int] = None
    local: Optional[str] = None

    @classmethod
    def parse(cls, v: str) -> "Version":
        m = SEMVER_RE.match(v.strip())
        if not m:
            raise ValueError(f"Invalid version: {v}")
        return cls(
            mjr=int(m.group(1)),
            mnr=int(m.group(2)),
            patch=int(m.group(3)),
            pre=Pre(m.group(4) or "release"),
            n=int(m.group(5)) if m.group(5) else None,
            local=m.group(6) if m.group(6) else None,
        )

    def __str__(self):
        base = f"{self.mjr}.{self.mnr}.{self.patch}"
        if self.pre == Pre.RELEASE:
            return base
        if self.local:
            return f"{base}-{self.pre}.{self.n}+{self.local}"
        return f"{base}-{self.pre}.{self.n}"

    def bump_base(self, level: str):
        if level == "none":
            return self

        if level == "major":
            self.mjr += 1
            self.mnr = 0
            self.patch = 0
        elif level == "minor":
            self.mnr += 1
            self.patch = 0
        elif level == "patch":
            self.patch += 1
        else:
            raise ValueError(f"Unknown bump level: {level}")

        # reset prerelease
        self.pre = Pre.RELEASE
        self.n = None
        self.local = None
        return self

    def bump_prerelease(self, target: Pre, *, local: Optional[str]):
        if target == Pre.RELEASE:
            # strip prerelease
            self.pre = Pre.RELEASE
            self.n = None
            self.local = None
            return self

        # entering prerelease mode
        if self.pre == Pre.RELEASE:
            self.pre = target
            self.n = 0
            self.local = local
            return self

        # switching prerelease stage
        if target != self.pre:
            self.pre = target
            self.n = 0
            self.local = local
        else:
            # bump n within same prerelease tier
            self.n += 1

        return self

# ------------------ File discovery / update ---------------------------------

def find_version_files(start: Path) -> List[Path]:
    files = []
    for p in start.rglob("*"):
        if p.name in ("pyproject.toml", "Cargo.toml"):
            files.append(p)
    return files

def read_version(pyproject: Path) -> Version:
    data = toml.load(pyproject)
    return Version.parse(data["project"]["version"])

def write_version(path: Path, version: Version):
    data = toml.load(path)
    if path.name == "pyproject.toml":
        data["project"]["version"] = str(version)
    else:
        data["package"]["version"] = str(version)
    with open(path, "w") as f:
        toml.dump(data, f)

# ----------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release_level", choices=["none", "patch", "minor", "major"])
    ap.add_argument("prerelease", choices=["alpha", "beta", "rc", "release"])
    ap.add_argument("--local", help="Local metadata (for alpha)", default=None)
    ap.add_argument("--start", default=".")
    args = ap.parse_args()

    root = Path(args.start) / "pyproject.toml"
    version = read_version(root)

    # base bump
    version.bump_base(args.release_level)

    # prerelease bump
    version.bump_prerelease(Pre(args.prerelease), local=args.local)

    # sync files
    for f in find_version_files(Path(args.start)):
        write_version(f, version)

    print(str(version))

if __name__ == "__main__":
    main()
