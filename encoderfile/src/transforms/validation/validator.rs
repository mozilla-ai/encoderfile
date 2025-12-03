use anyhow::Result;
use encoderfile_core::transforms::Transform;

pub trait TransformValidator {
    fn validate(&self, transform: Transform) -> Result<()>;
}
