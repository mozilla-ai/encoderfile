use std::marker::PhantomData;

use crate::{
    common::model_type::ModelTypeSpec, format::footer::EncoderfileFooter,
    generated::manifest::EncoderfileManifest,
};

pub mod assets;
pub mod codec;
pub mod footer;

pub struct Encoderfile {
    manifest: EncoderfileManifest,
    footer: EncoderfileFooter,
}
