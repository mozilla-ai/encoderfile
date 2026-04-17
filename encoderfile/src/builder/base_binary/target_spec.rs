use super::error::TargetError;
use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, Hash, JsonSchema)]
pub struct TargetSpec {
    pub arch: Architecture,
    pub os: OperatingSystem,
    pub abi: Abi,
}

impl Serialize for TargetSpec {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for TargetSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl FromStr for TargetSpec {
    type Err = TargetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('-').collect();

        match parts.as_slice() {
            // ---------- Linux ----------
            [arch, "unknown", "linux", abi] => {
                let arch = match *arch {
                    "x86_64" => Architecture::X86_64,
                    "aarch64" => Architecture::Aarch64,
                    other => {
                        return Err(TargetError::UnsupportedArchForOs {
                            arch: other.to_string(),
                            os: "linux",
                        });
                    }
                };

                let abi = match *abi {
                    "gnu" => Abi::Gnu,
                    "musl" => Abi::Musl,
                    other => {
                        return Err(TargetError::UnsupportedAbi {
                            abi: other.to_string(),
                            os: "linux",
                        });
                    }
                };

                Ok(Self {
                    arch,
                    os: OperatingSystem::Linux,
                    abi,
                })
            }

            // ---------- macOS ----------
            [arch, "apple", "darwin"] => {
                let arch = match *arch {
                    "x86_64" => Architecture::X86_64,
                    "aarch64" => Architecture::Aarch64,
                    other => {
                        return Err(TargetError::UnsupportedArchForOs {
                            arch: other.to_string(),
                            os: "apple",
                        });
                    }
                };

                Ok(Self {
                    arch,
                    os: OperatingSystem::MacOS,
                    abi: Abi::Gnu, // placeholder, not used on macOS
                })
            }

            // ---------- Windows ----------
            [arch, "pc", "windows", "msvc"] => {
                let arch = match *arch {
                    "x86_64" => Architecture::X86_64,
                    other => {
                        return Err(TargetError::UnsupportedArchForOs {
                            arch: other.to_string(),
                            os: "windows",
                        });
                    }
                };

                Ok(Self {
                    arch,
                    os: OperatingSystem::Windows,
                    abi: Abi::Msvc,
                })
            }

            _ => Err(TargetError::InvalidTriple(s.to_string())),
        }
    }
}

impl TargetSpec {
    pub fn detect_host() -> Result<Self, TargetError> {
        let arch = match std::env::consts::ARCH {
            "x86_64" => Architecture::X86_64,
            "aarch64" => Architecture::Aarch64,
            other => return Err(TargetError::UnsupportedArch(other.to_string())),
        };

        let os = match std::env::consts::OS {
            "linux" => OperatingSystem::Linux,
            "macos" => OperatingSystem::MacOS,
            "windows" => OperatingSystem::Windows,
            other => return Err(TargetError::UnsupportedOs(other.to_string())),
        };

        let abi = match os {
            OperatingSystem::Linux => Abi::Gnu,
            OperatingSystem::MacOS => Abi::Gnu, // unused but harmless
            OperatingSystem::Windows => Abi::Msvc,
        };

        Ok(Self { arch, os, abi })
    }
}

impl std::fmt::Display for TargetSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.os {
            OperatingSystem::Linux => write!(f, "{}-unknown-linux-{}", self.arch, self.abi),
            OperatingSystem::MacOS => write!(f, "{}-apple-darwin", self.arch),
            OperatingSystem::Windows => write!(f, "{}-pc-windows-msvc", self.arch),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, JsonSchema)]
pub enum Architecture {
    X86_64,
    Aarch64,
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::X86_64 => write!(f, "x86_64"),
            Self::Aarch64 => write!(f, "aarch64"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, JsonSchema)]
pub enum OperatingSystem {
    Linux,
    MacOS,
    Windows,
}

impl std::fmt::Display for OperatingSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linux => write!(f, "linux"),
            Self::MacOS => write!(f, "darwin"),
            Self::Windows => write!(f, "windows"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, JsonSchema)]
pub enum Abi {
    Gnu,
    Musl,
    Msvc,
}

impl std::fmt::Display for Abi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gnu => write!(f, "gnu"),
            Self::Musl => write!(f, "musl"),
            Self::Msvc => write!(f, "msvc"),
        }
    }
}
