"""
Bump semantic version
"""

from typing import Optional, List, Tuple
import sys
import os
import re
import argparse
from enum import StrEnum
from pathlib import Path
import toml
from pydantic import BaseModel

PATTERN = re.compile(
    r"""
v?(\d+)\.(\d+)\.(\d+)(?:-(alpha|beta|rc)\.(\d+)(?:\+(\w+))?)?
""".strip()
)


class Pre(StrEnum):
    ALPHA = "alpha"
    BETA = "beta"
    RC = "rc"
    RELEASE = "release"

    def __int__(self) -> int:
        if self == Pre.ALPHA:
            return 0
        if self == Pre.BETA:
            return 1
        if self == Pre.RC:
            return 2
        if self == Pre.RELEASE:
            return 3
    
    def __gt__(self, other: str) -> bool:
        return int(self) > int(other)
    
    def __lt__(self, other: str) -> bool:
        return int(self) < int(other)

    def __ge__(self, other: str) -> bool:
        return int(self) >= int(other)
    
    def __le__(self, other: str) -> bool:
        return int(self) <= int(other)

class Version(BaseModel):
    """Version model."""

    mjr: int
    mnr: int
    patch: int
    pre: Pre = Pre.RELEASE
    n: Optional[int] = None
    local: Optional[str] = None

    @property
    def base_version(self) -> str:
        """Base version."""
        return f"{self.mjr}.{self.mnr}.{self.patch}"

    @classmethod
    def parse(cls, version: str) -> "Version":
        """Parse string."""
        match = PATTERN.match(version.strip())
        if not match:
            raise ValueError(f"Invalid version: {version}")
        return cls(
            mjr=int(match.group(1)),
            mnr=int(match.group(2)),
            patch=int(match.group(3)),
            pre=Pre(match.group(4) or "release"),
            n=int(match.group(5)) if match.group(5) else None,
            local=match.group(6) if match.group(6) else None,
        )

    def __str__(self) -> str:
        if self.local is not None:
            return f"{self.base_version}-{self.pre}.{self.n}+{self.local}"
        if self.n is not None:
            return f"{self.base_version}-{self.pre}.{self.n}"
        if self.pre != Pre.RELEASE:
            return f"{self.base_version}-{self.pre}"

        return self.base_version

    def bump_pre(self, target: Pre, local: Optional[str] = None):
        # Can't go backwards without bumping version
        if target < self.pre:
            raise ValueError(
                f"Cannot move from {self.pre} to {target} without bumping base version"
            )

        # alpha always bumps counter
        if target == Pre.ALPHA:
            self.local = local
            if self.n is None:
                self.n = 0
            self.n += 1

        # beta or rc should preserve `n`, but switch to 0 if absent if alpha isn't used
        elif target in {Pre.BETA, Pre.RC}:
            if self.n is None:
                self.n = 0
            self.local = None

        # release drops counter & local
        elif target == Pre.RELEASE:
            self.n = None
            self.local = None

        self.pre = target
        return self


    def bump_dev(self, local: str) -> "Version":
        """Bump dev version."""
        return self.bump_pre(Pre.ALPHA, local)

    def bump_beta(self) -> "Version":
        """Bump beta version."""
        return self.bump_pre(Pre.BETA)

    def bump_rc(self) -> "Version":
        """Bump rc version."""
        return self.bump_pre(Pre.RC)

    def bump_release(self) -> "Version":
        """Bump release version."""
        return self.bump_pre(Pre.RELEASE)

    def bump_mjr(self) -> "Version":
        """Bump major version."""
        if self.pre != Pre.RELEASE:
            raise ValueError(
                "You have to have a release version before creating a mjr version"
            )
        self.mjr += 1
        self.mnr = 0
        self.patch = 0
        self.pre = Pre.RELEASE
        self.n = None
        self.local = None
        return self

    def bump_mnr(self) -> "Version":
        """Bump minor version."""
        if self.pre != Pre.RELEASE:
            raise ValueError(
                "You have to have a release version before creating a mnr version"
            )
        self.mnr += 1
        self.patch = 0
        self.pre = Pre.RELEASE
        self.n = None
        self.local = None
        return self

    def bump_patch(self) -> "Version":
        """Bump patch version."""
        if self.pre != Pre.RELEASE:
            raise ValueError(
                "You have to have a release version before creating a patch version"
            )
        self.patch += 1
        self.pre = Pre.RELEASE
        self.n = None
        self.local = None
        return self


def find_version_files(start_path: str = ".") -> List[Path]:
    """Recursively find all pyproject.toml and Cargo.toml files except root pyproject.toml."""
    version_files = []
    start_path = Path(start_path)
    root_pyproject = start_path / "pyproject.toml"

    # Find all files except the root pyproject.toml
    for path in start_path.rglob("*"):
        if path.name in ["pyproject.toml", "Cargo.toml"] and path != root_pyproject:
            version_files.append(path)

    return version_files


def get_root_version(pyproject_path: Path) -> Version:
    """Get version from root pyproject.toml."""
    try:
        with open(pyproject_path, "r", encoding="utf-8") as f:
            data = toml.load(f)

        if "project" not in data or "version" not in data["project"]:
            raise ValueError("No project.version field found in root pyproject.toml")

        return Version.parse(data["project"]["version"])
    except FileNotFoundError as e:
        raise ValueError(f"Root pyproject.toml not found at {pyproject_path}") from e
    except Exception as e:
        raise ValueError(f"Error reading root version: {str(e)}") from e


def update_version_in_file(file_path: Path, new_version: Version) -> Tuple[bool, str]:
    """Update version in a TOML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = toml.load(f)

        # Determine the correct path to version based on file type
        if file_path.name == "pyproject.toml":
            if "project" not in data or "version" not in data["project"]:
                return False, "No project.version field found"
            data["project"]["version"] = str(new_version)
        else:  # Cargo.toml
            if "package" not in data or "version" not in data["package"]:
                return False, "No package.version field found"
            data["package"]["version"] = str(new_version)

        # Write the updated TOML back to file
        with open(file_path, "w", encoding="utf-8") as f:
            toml.dump(data, f)

        return True, f"Updated version to {new_version}"
    except Exception as e:
        return False, f"Error updating file: {str(e)}"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Bump version numbers according to semantic versioning and sync across all files"
    )

    parser.add_argument(
        "action",
        choices=["alpha", "beta", "rc", "release", "mjr", "mnr", "patch"],
        help="The type of version bump to perform",
    )

    parser.add_argument(
        "--start-path",
        default=".",
        help="Path to root directory containing pyproject.toml (default: current directory)",
    )

    parser.add_argument(
        "--git-hash",
        help="Git hash to use for alpha versions",
    )

    args = parser.parse_args()
    start_path = Path(args.start_path)
    root_pyproject = start_path / "pyproject.toml"

    try:
        # First, get and bump the root version
        version = get_root_version(root_pyproject)
        original_version = str(version)  # Store for logging

        # Apply the requested version bump
        try:
            if args.action == "alpha":
                if not args.git_hash:
                    raise ValueError("Git hash is required for alpha versions")
                version.bump_dev(args.git_hash)
            elif args.action == "beta":
                version.bump_beta()
            elif args.action == "rc":
                version.bump_rc()
            elif args.action == "release":
                version.bump_release()
            elif args.action == "mjr":
                version.bump_mjr()
            elif args.action == "mnr":
                version.bump_mnr()
            elif args.action == "patch":
                version.bump_patch()
        except ValueError as e:
            print(f"Error bumping version: {e}")
            return 1

        # Update root pyproject.toml first
        success, message = update_version_in_file(root_pyproject, version)
        if not success:
            print(f"Failed to update root pyproject.toml: {message}")
            return 1

        print(f"✓ Bumped version from {original_version} to {version}")
        print(f"✓ Updated root {root_pyproject}")

        # Find and update all other version files
        version_files = find_version_files(args.start_path)
        if not version_files:
            print("No additional version files found to sync")
            return 0

        print("\nSyncing version to other files:")
        for file_path in version_files:
            # don't update root pyproject.toml
            if os.path.abspath(file_path) == os.path.abspath(root_pyproject):
                continue

            if os.path.abspath(file_path) == os.path.abspath("./Cargo.toml"):
                continue

            # don't update files in .venv
            if ".venv" in str(file_path):
                continue

            success, message = update_version_in_file(file_path, version)
            if success:
                print(f"✓ {file_path}: {message}")
            else:
                print(f"✗ {file_path}: {message}")

    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
