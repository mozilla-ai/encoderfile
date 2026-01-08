// Manifest protobufs

tonic::include_proto!("encoderfile.manifest");

impl Artifact {
    pub fn new(offset: u64, length: u64, sha256: [u8; 32]) -> Artifact {
        Artifact {
            offset,
            length,
            sha256: sha256.to_vec()
        }
    }
}
