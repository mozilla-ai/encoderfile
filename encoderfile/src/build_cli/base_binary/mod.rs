use crate::build_cli::terminal;
use anyhow::{Context, Result, bail};
use std::{
    fs,
    path::{Path, PathBuf},
};
use url::Url;

#[cfg(test)]
use std::cell::RefCell;

pub mod target_spec;

pub use target_spec::{Abi, Architecture, OperatingSystem, TargetSpec};

const DOWNLOAD_URL: &str = "https://github.com/mozilla-ai/encoderfile/releases/download/";
const DOWNLOAD_URL_OVERRIDE_ENV_VAR: &str = "ENCODERFILE_BASE_BINARY_BASE_URL";
const ENCODERFILE_RUNTIME_NAME: &str = "encoderfile-runtime";

#[derive(Debug)]
pub struct BaseBinaryResolver<'a> {
    pub cache_dir: &'a Path,
    pub base_binary_path: Option<&'a Path>,
    pub target: TargetSpec,
    pub version: Option<String>,
}

impl BaseBinaryResolver<'_> {
    pub fn remove(&self) -> Result<()> {
        // Explicit path overrides cache semantics
        if self.base_binary_path.is_some() {
            anyhow::bail!("cannot remove an explicitly provided base binary path");
        }

        let path = self.cache_path();

        if !path.exists() {
            // idempotent: removing something that isn't there is fine
            return Ok(());
        }

        fs::remove_file(&path).with_context(|| format!("failed to remove {}", path.display()))?;

        self.cleanup_empty_parents(path.parent());

        Ok(())
    }

    fn cleanup_empty_parents(&self, mut dir: Option<&Path>) {
        while let Some(d) = dir {
            // Stop at cache_dir — never go above it
            if d == self.cache_dir {
                break;
            }

            match fs::read_dir(d) {
                Ok(mut entries) => {
                    if entries.next().is_none() {
                        let _ = fs::remove_dir(d);
                        dir = d.parent();
                    }
                }
                _ => break,
            }
        }
    }

    pub fn resolve(&self, no_download: bool) -> Result<PathBuf> {
        // 1. Explicit override always wins
        if let Some(path) = self.base_binary_path {
            terminal::info_kv("Using local binary target:", path.display());

            return Ok(path.to_path_buf());
        }

        let final_path = self.cache_path();

        // 2. Cache hit
        if final_path.exists() {
            terminal::success("Binary already cached");
            self.validate_binary(&final_path)?;
            return Ok(final_path);
        }

        // 3. Cache miss → download
        match no_download {
            false => self.download_and_install(&final_path)?,
            true => bail!("Cannot download {:?}", self.file_name()),
        }

        // 4. Final sanity check
        self.validate_binary(&final_path)?;

        Ok(final_path)
    }

    fn cache_path(&self) -> PathBuf {
        self.cache_dir
            .join("base-binaries")
            .join("encoderfile")
            .join(self.version())
            .join(self.target.to_string())
            .join(ENCODERFILE_RUNTIME_NAME)
    }

    fn validate_binary(&self, path: &Path) -> Result<()> {
        terminal::info("Validating binary...");
        use std::os::unix::fs::PermissionsExt;

        let meta = fs::metadata(path)
            .with_context(|| format!("base binary missing at {}", path.display()))?;

        if !meta.is_file() {
            anyhow::bail!("base binary is not a file: {}", path.display());
        }

        let mode = meta.permissions().mode();
        if mode & 0o111 == 0 {
            anyhow::bail!("base binary is not executable: {}", path.display());
        }

        terminal::success("Binary validated");

        Ok(())
    }

    fn download_and_install(&self, final_path: &Path) -> Result<()> {
        use std::io;
        use tempfile::{NamedTempFile, TempDir};

        terminal::info("Base binary not found locally. Downloading...");

        let url = self.download_url()?;

        let parent = final_path.parent().expect("cache path always has a parent");

        fs::create_dir_all(parent)?;

        // ---- download to temp file ----
        let mut resp = reqwest::blocking::get(url.as_str())
            .with_context(|| format!("failed to download {}", url))?;

        if !resp.status().is_success() {
            anyhow::bail!("download failed with status {} for {}", resp.status(), url);
        }

        let mut archive = NamedTempFile::new_in(self.cache_dir)?;
        io::copy(&mut resp, &mut archive)?;

        // ---- extract to temp dir ----
        let extract_dir = TempDir::new_in(self.cache_dir)?;
        let tar_gz = fs::File::open(archive.path())?;
        let decoder = flate2::read::GzDecoder::new(tar_gz);
        let mut archive = tar::Archive::new(decoder);

        archive.unpack(&extract_dir)?;

        // ---- move runtime into final place ----
        let extracted = extract_dir.path().join(ENCODERFILE_RUNTIME_NAME);

        if !extracted.exists() {
            anyhow::bail!("archive did not contain `{}`", ENCODERFILE_RUNTIME_NAME);
        }

        fs::rename(extracted, final_path)?;

        terminal::success("Binary successfully downloaded");

        Ok(())
    }
}

impl BaseBinaryResolver<'_> {
    fn download_url(&self) -> Result<Url> {
        let version = self.version();
        let file_name = self.file_name();

        self.base_url()?
            .join(&format!("v{}/", version))?
            .join(&file_name)
            .context("Failed to construct download url")
    }

    fn version(&self) -> String {
        self.version
            .clone()
            .unwrap_or(env!("CARGO_PKG_VERSION").to_string())
    }

    pub fn file_name(&self) -> String {
        format!("{ENCODERFILE_RUNTIME_NAME}-{}.tar.gz", self.target)
    }

    fn base_url(&self) -> Result<Url> {
        if let Some(raw) = base_url_override() {
            let mut url = Url::parse(&raw).map_err(|e| {
                anyhow::anyhow!("invalid {DOWNLOAD_URL_OVERRIDE_ENV_VAR} `{raw}`: {e}")
            })?;

            if !url.as_str().ends_with('/') {
                url = Url::parse(&(url.as_str().to_owned() + "/"))?;
            }

            Ok(url)
        } else {
            Ok(Url::parse(DOWNLOAD_URL)?)
        }
    }
}

#[derive(Debug)]
pub struct DownloadedRuntime {
    pub target: TargetSpec,
    pub version: String,
    pub path: PathBuf,
}

pub fn list_downloaded_runtimes(cache_dir: &Path) -> Result<Vec<DownloadedRuntime>> {
    let mut results = Vec::new();

    let root = cache_dir.join("base-binaries").join("encoderfile");

    if !root.exists() {
        return Ok(results);
    }

    for version_entry in fs::read_dir(&root)? {
        let version_entry = version_entry?;
        if !version_entry.file_type()?.is_dir() {
            continue;
        }

        let version = version_entry.file_name().to_string_lossy().to_string();
        let version_dir = version_entry.path();

        for target_entry in fs::read_dir(&version_dir)? {
            let target_entry = target_entry?;
            if !target_entry.file_type()?.is_dir() {
                continue;
            }

            let target_entry_file_name = target_entry.file_name();

            let target_str = target_entry_file_name.to_string_lossy();
            let target: TargetSpec = match target_str.parse() {
                Ok(t) => t,
                Err(_) => continue, // skip unknown junk
            };

            let runtime_path = target_entry.path().join(ENCODERFILE_RUNTIME_NAME);

            if runtime_path.exists() {
                results.push(DownloadedRuntime {
                    target,
                    version: version.clone(),
                    path: runtime_path,
                });
            }
        }
    }

    Ok(results)
}

// -------- env access seam --------

#[cfg(not(test))]
fn base_url_override() -> Option<String> {
    std::env::var(DOWNLOAD_URL_OVERRIDE_ENV_VAR).ok()
}

#[cfg(test)]
fn base_url_override() -> Option<String> {
    TEST_BASE_URL.with(|v| v.borrow().clone())
}

// -------- thread-local override --------

#[cfg(test)]
thread_local! {
    static TEST_BASE_URL: RefCell<Option<String>> = const { RefCell::new(None) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_base_url(value: Option<&str>, f: impl FnOnce()) {
        TEST_BASE_URL.with(|v| {
            *v.borrow_mut() = value.map(|s| s.to_string());
        });
        f();
        TEST_BASE_URL.with(|v| {
            *v.borrow_mut() = None;
        });
    }

    // -------- helpers --------

    impl<'a> BaseBinaryResolver<'a> {
        fn test_new(target: TargetSpec, version: Option<&'a str>) -> Self {
            Self {
                cache_dir: Path::new("/tmp"),
                base_binary_path: None,
                target,
                version: version.map(|i| i.to_string()),
            }
        }
    }

    fn target() -> TargetSpec {
        "x86_64-unknown-linux-gnu".parse().unwrap()
    }

    // ---------- version() ----------

    #[test]
    fn version_uses_override_when_present() {
        let r = BaseBinaryResolver::test_new(target(), Some("1.2.3"));
        assert_eq!(r.version(), "1.2.3");
    }

    #[test]
    fn version_defaults_to_crate_version() {
        let r = BaseBinaryResolver::test_new(target(), None);
        assert_eq!(r.version(), env!("CARGO_PKG_VERSION"));
    }

    // ---------- file_name() ----------

    #[test]
    fn file_name_is_canonical() {
        let r = BaseBinaryResolver::test_new(target(), None);
        assert_eq!(
            r.file_name(),
            "encoderfile-runtime-x86_64-unknown-linux-gnu.tar.gz"
        );
    }

    // ---------- base_url() ----------

    #[test]
    fn base_url_defaults_to_github() {
        with_base_url(None, || {
            let r = BaseBinaryResolver::test_new(target(), None);
            assert_eq!(
                r.base_url().unwrap().as_str(),
                "https://github.com/mozilla-ai/encoderfile/releases/download/"
            );
        });
    }

    #[test]
    fn base_url_uses_override() {
        with_base_url(Some("https://example.com/releases/"), || {
            let r = BaseBinaryResolver::test_new(target(), None);
            assert_eq!(
                r.base_url().unwrap().as_str(),
                "https://example.com/releases/"
            );
        });
    }

    #[test]
    fn base_url_adds_trailing_slash_if_missing() {
        with_base_url(Some("https://example.com/releases"), || {
            let r = BaseBinaryResolver::test_new(target(), None);
            assert_eq!(
                r.base_url().unwrap().as_str(),
                "https://example.com/releases/"
            );
        });
    }

    #[test]
    fn base_url_rejects_invalid_url() {
        with_base_url(Some("not a url"), || {
            let r = BaseBinaryResolver::test_new(target(), None);
            assert!(r.base_url().is_err());
        });
    }

    // ---------- download_url() ----------

    #[test]
    fn download_url_is_correct() {
        with_base_url(None, || {
            let r = BaseBinaryResolver::test_new(target(), Some("0.3.1"));
            let url = r.download_url().unwrap();

            assert_eq!(
                url.as_str(),
                "https://github.com/mozilla-ai/encoderfile/releases/download/v0.3.1/encoderfile-runtime-x86_64-unknown-linux-gnu.tar.gz"
            );
        });
    }
}
