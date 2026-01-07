use anyhow::{Result, bail};
use std::io::{Read, Seek, SeekFrom, Write};

// flag 0: whether metadata is protobuf
pub const FLAG_METADATA_PROTOBUF: u32 = 1 << 0;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EncoderfileFooter {
    pub magic: [u8; 8],
    pub format_version: u32,
    pub metadata_offset: u64,
    pub metadata_length: u64,
    pub flags: u32,
}

impl EncoderfileFooter {
    pub const MAGIC: [u8; 8] = *b"ENCFILE\0";
    pub const SIZE: usize = 32;

    pub fn write_to<W: Write>(&self, mut w: W) -> Result<()> {
        w.write_all(&self.magic)?;
        w.write_all(&self.format_version.to_le_bytes())?;
        w.write_all(&self.metadata_offset.to_le_bytes())?;
        w.write_all(&self.metadata_length.to_le_bytes())?;
        w.write_all(&self.flags.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from<R: Read + Seek>(mut r: R) -> Result<Self> {
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
