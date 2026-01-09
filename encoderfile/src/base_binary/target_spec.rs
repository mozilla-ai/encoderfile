use anyhow::Result;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TargetSpec {
    pub arch: Architecture,
    pub os: OperatingSystem,
    pub abi: Abi,
}

impl FromStr for TargetSpec {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('-').collect();

        if parts.len() != 4 {
            anyhow::bail!("invalid target triple `{s}` (expected <arch>-<vendor>-<os>-<abi>)");
        }

        let arch = match parts[0] {
            "x86_64" => Architecture::X86_64,
            "aarch64" => Architecture::Aarch64,
            other => anyhow::bail!("unsupported architecture `{other}`"),
        };

        // parts[1] = vendor (ignored for now, must be "unknown")
        match parts[1] {
            "unknown" => {}
            other => anyhow::bail!("unsupported vendor `{other}`"),
        }

        let os = match parts[2] {
            "linux" => OperatingSystem::Linux,
            other => anyhow::bail!("unsupported operating system `{other}`"),
        };

        let abi = match parts[3] {
            "gnu" => Abi::Gnu,
            "musl" => Abi::Musl,
            other => anyhow::bail!("unsupported ABI `{other}`"),
        };

        Ok(Self { arch, os, abi })
    }
}

impl TargetSpec {
    pub fn detect_host() -> Result<Self> {
        let arch = match std::env::consts::ARCH {
            "x86_64" => Architecture::X86_64,
            "aarch64" => Architecture::Aarch64,
            other => anyhow::bail!("unsupported architecture: {other}"),
        };

        let os = match std::env::consts::OS {
            "linux" => OperatingSystem::Linux,
            other => anyhow::bail!("unsupported operating system: {other}"),
        };

        // For now, default to gnu
        // Later this can probe libc or be overridden by flags
        let abi = Abi::Gnu;

        Ok(Self { arch, os, abi })
    }
}

impl std::fmt::Display for TargetSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let arch = match self.arch {
            Architecture::X86_64 => "x86_64",
            Architecture::Aarch64 => "aarch64",
        };

        let abi = match self.abi {
            Abi::Gnu => "gnu",
            Abi::Musl => "musl",
        };

        let os = match self.os {
            OperatingSystem::Linux => "unknown-linux",
        };

        write!(f, "{arch}-{os}-{abi}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    X86_64,
    Aarch64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatingSystem {
    Linux,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Abi {
    Gnu,
    Musl,
}
