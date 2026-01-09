use anyhow::{Context, Result};
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
    cache_dir: &'a Path,
    base_binary_path: Option<&'a Path>,
    target: TargetSpec,
    version: Option<&'a str>,
}

impl BaseBinaryResolver<'_> {
    pub fn resolve(&self) -> Result<PathBuf> {
        // 1. Explicit override always wins
        if let Some(path) = self.base_binary_path {
            return Ok(path.to_path_buf());
        }

        let final_path = self.cache_path();

        // 2. Cache hit
        if final_path.exists() {
            self.validate_binary(&final_path)?;
            return Ok(final_path);
        }

        // 3. Cache miss → download
        self.download_and_install(&final_path)?;

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

        Ok(())
    }

    fn download_and_install(&self, final_path: &Path) -> Result<()> {
        use std::io;
        use tempfile::{NamedTempFile, TempDir};

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

        Ok(())
    }
}

impl BaseBinaryResolver<'_> {
    fn download_url(&self) -> Result<Url> {
        let version = self.version();
        let file_name = self.file_name();

        self.base_url()?
            .join(&format!("{}/", version))?
            .join(&file_name)
            .context("Failed to construct download url")
    }

    fn version(&self) -> &str {
        self.version.unwrap_or(env!("CARGO_PKG_VERSION"))
    }

    fn file_name(&self) -> String {
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
    static TEST_BASE_URL: RefCell<Option<String>> = RefCell::new(None);
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
                version,
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
                "https://github.com/mozilla-ai/encoderfile/releases/download/0.3.1/encoderfile-runtime-x86_64-unknown-linux-gnu.tar.gz"
            );
        });
    }
}
