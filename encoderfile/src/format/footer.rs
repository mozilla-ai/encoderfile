use anyhow::{Result, bail};
use std::io::{Read, Seek, SeekFrom, Write};

// flag 0: whether metadata is protobuf
pub const FLAG_METADATA_PROTOBUF: u32 = 1 << 0;

#[repr(C)]
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct EncoderfileFooter {
    pub magic: [u8; 8],
    pub format_version: u32,
    pub metadata_offset: u64,
    pub metadata_length: u64,
    pub flags: u32,
}

impl EncoderfileFooter {
    pub const MAGIC: [u8; 8] = *b"ENCFILE\0";
    pub const CURRENT_VERSION: u32 = 1;
    pub const SIZE: usize = 32;

    pub fn new(metadata_offset: u64, metadata_length: u64, metadata_is_protobuf: bool) -> Self {
        let mut flags = 0;

        if metadata_is_protobuf {
            flags |= FLAG_METADATA_PROTOBUF;
        }

        EncoderfileFooter {
            magic: Self::MAGIC,
            format_version: Self::CURRENT_VERSION,
            metadata_offset,
            metadata_length,
            flags,
        }
    }

    pub fn write_to<W: Write>(&self, mut w: W) -> Result<()> {
        w.write_all(&self.magic)?;
        w.write_all(&self.format_version.to_le_bytes())?;
        w.write_all(&self.metadata_offset.to_le_bytes())?;
        w.write_all(&self.metadata_length.to_le_bytes())?;
        w.write_all(&self.flags.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from<R: Read + Seek>(r: &mut R) -> Result<Self> {
        r.seek(SeekFrom::End(-(Self::SIZE as i64)))?;

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;

        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        let format_version = u32::from_le_bytes(buf);

        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)?;
        let metadata_offset = u64::from_le_bytes(buf);

        r.read_exact(&mut buf)?;
        let metadata_length = u64::from_le_bytes(buf);

        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        let flags = u32::from_le_bytes(buf);

        if magic != Self::MAGIC {
            bail!("Not a valid encoderfile.")
        }

        Ok(Self {
            magic,
            format_version,
            metadata_offset,
            metadata_length,
            flags,
        })
    }

    pub fn validate(self) -> Result<Self> {
        if self.magic != Self::MAGIC {
            bail!("Not a valid encoderfile.")
        }

        if self.format_version == 1 && (self.flags & FLAG_METADATA_PROTOBUF == 0) {
            bail!("format v1 requires protobuf metadata");
        }

        Ok(self)
    }

    pub fn has_flag(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Seek, SeekFrom};

    fn footer() -> EncoderfileFooter {
        EncoderfileFooter::new(123, 456, true)
    }

    #[test]
    fn new_sets_fields_and_flags() {
        let f = EncoderfileFooter::new(10, 20, true);

        assert_eq!(f.magic, EncoderfileFooter::MAGIC);
        assert_eq!(f.format_version, EncoderfileFooter::CURRENT_VERSION);
        assert_eq!(f.metadata_offset, 10);
        assert_eq!(f.metadata_length, 20);
        assert!(f.has_flag(FLAG_METADATA_PROTOBUF));
    }

    #[test]
    fn new_without_protobuf_flag() {
        let f = EncoderfileFooter::new(0, 0, false);
        assert!(!f.has_flag(FLAG_METADATA_PROTOBUF));
    }

    #[test]
    fn write_and_read_round_trip() {
        let f = footer();

        let mut buf = Vec::new();
        f.write_to(&mut buf).unwrap();

        // simulate file with footer at end
        let mut cursor = Cursor::new(buf);
        cursor.seek(SeekFrom::End(0)).unwrap();

        let read = EncoderfileFooter::read_from(&mut cursor).unwrap();

        assert_eq!(read.magic, f.magic);
        assert_eq!(read.format_version, f.format_version);
        assert_eq!(read.metadata_offset, f.metadata_offset);
        assert_eq!(read.metadata_length, f.metadata_length);
        assert_eq!(read.flags, f.flags);
    }

    #[test]
    fn read_fails_on_bad_magic() {
        let mut buf = Vec::new();

        let mut f = footer();
        f.magic = *b"BADMAGIC";
        f.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(buf);
        cursor.seek(SeekFrom::End(0)).unwrap();

        let err = EncoderfileFooter::read_from(&mut cursor).unwrap_err();
        assert!(err.to_string().contains("Not a valid encoderfile"));
    }

    #[test]
    fn validate_accepts_valid_footer() {
        let f = footer();
        f.validate().unwrap();
    }

    #[test]
    fn validate_rejects_missing_protobuf_flag_for_v1() {
        let f = EncoderfileFooter {
            flags: 0, // no protobuf flag
            ..footer()
        };

        let err = f.validate().unwrap_err();
        assert!(err.to_string().contains("requires protobuf metadata"));
    }

    #[test]
    fn has_flag_works() {
        let f = footer();

        assert!(f.has_flag(FLAG_METADATA_PROTOBUF));
        assert!(!f.has_flag(1 << 31));
    }

    #[test]
    fn read_fails_on_truncated_input() {
        let mut buf = vec![0u8; EncoderfileFooter::SIZE - 1];
        let mut cursor = Cursor::new(&mut buf);

        let err = EncoderfileFooter::read_from(&mut cursor).unwrap_err();
        let msg = err.to_string();

        assert!(
            msg.contains("failed") || msg.contains("seek") || msg.contains("read"),
            "unexpected error: {msg}"
        );
    }
}
