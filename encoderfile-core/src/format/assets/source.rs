use anyhow::Result;
use ring::digest;
use std::{
    borrow::Cow,
    fs::File,
    io::{BufReader, Cursor, Read},
    path::Path,
};

#[derive(Debug)]
pub enum AssetSource<'a> {
    File(&'a Path),
    InMemory(Cow<'a, [u8]>),
}

impl<'a> AssetSource<'a> {
    pub fn write_to<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<u64> {
        match self {
            AssetSource::File(path) => {
                let file = File::open(path)?;
                let mut reader = BufReader::new(file);
                std::io::copy(&mut reader, out)
            }
            AssetSource::InMemory(bytes) => {
                out.write_all(bytes)?;
                Ok(bytes.len() as u64)
            }
        }
    }

    fn open(&'a self) -> std::io::Result<Box<dyn Read + 'a>> {
        match self {
            AssetSource::File(path) => Ok(Box::new(File::open(path)?)),
            AssetSource::InMemory(bytes) => Ok(Box::new(Cursor::new(bytes.as_ref()))),
        }
    }

    pub fn hash_and_len(&self) -> Result<(u64, [u8; 32])> {
        let reader = self.open()?;
        let mut reader = BufReader::new(reader);

        let mut ctx = digest::Context::new(&digest::SHA256);
        let mut buf = [0u8; 64 * 1024];
        let mut len: u64 = 0;

        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            ctx.update(&buf[..n]);
            len += n as u64;
        }

        let digest = ctx.finish();
        let mut out = [0u8; 32];
        out.copy_from_slice(digest.as_ref());

        Ok((len, out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        borrow::Cow,
        fs,
        path::PathBuf,
    };

    fn sha256(bytes: &[u8]) -> [u8; 32] {
        use ring::digest;
        let digest = digest::digest(&digest::SHA256, bytes);
        let mut out = [0u8; 32];
        out.copy_from_slice(digest.as_ref());
        out
    }

    fn temp_file_with_contents(bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("asset_test_{}.bin", uuid::Uuid::new_v4()));
        fs::write(&path, bytes).expect("failed to write temp file");
        path
    }

    #[test]
    fn in_memory_hash_and_len() {
        let data = b"hello world";
        let asset = AssetSource::InMemory(Cow::Borrowed(data));

        let (len, hash) = asset.hash_and_len().unwrap();

        assert_eq!(len, data.len() as u64);
        assert_eq!(hash, sha256(data));
    }

    #[test]
    fn file_hash_and_len() {
        let data = b"hello world";
        let path = temp_file_with_contents(data);
        let asset = AssetSource::File(&path);

        let (len, hash) = asset.hash_and_len().unwrap();

        assert_eq!(len, data.len() as u64);
        assert_eq!(hash, sha256(data));

        fs::remove_file(path).ok();
    }

    #[test]
    fn file_and_memory_match() {
        let data = b"same bytes, different source";
        let path = temp_file_with_contents(data);

        let mem = AssetSource::InMemory(Cow::Borrowed(data));
        let file = AssetSource::File(&path);

        let mem_res = mem.hash_and_len().unwrap();
        let file_res = file.hash_and_len().unwrap();

        assert_eq!(mem_res, file_res);

        fs::remove_file(path).ok();
    }

    #[test]
    fn write_to_in_memory() {
        let data = b"write me";
        let asset = AssetSource::InMemory(Cow::Borrowed(data));

        let mut out = Vec::new();
        let written = asset.write_to(&mut out).unwrap();

        assert_eq!(written, data.len() as u64);
        assert_eq!(out, data);
    }

    #[test]
    fn write_to_file() {
        let data = b"file write test";
        let path = temp_file_with_contents(data);
        let asset = AssetSource::File(&path);

        let mut out = Vec::new();
        let written = asset.write_to(&mut out).unwrap();

        assert_eq!(written, data.len() as u64);
        assert_eq!(out, data);

        fs::remove_file(path).ok();
    }
}
