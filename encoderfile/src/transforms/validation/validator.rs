use anyhow::Result;
use encoderfile_core::transforms::Transform;

pub trait TransformValidator {
    fn validate_transform(&self, transform: Transform) -> Result<()>;
}
