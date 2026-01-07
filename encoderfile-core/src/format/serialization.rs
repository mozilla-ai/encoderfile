use std::marker::PhantomData;

use crate::{
    common::model_type::ModelTypeSpec, format::footer::EncoderfileFooter,
    generated::manifest::EncoderfileManifest,
};

#[derive(Debug)]
pub struct EncoderfileSerializer<T: ModelTypeSpec> {
    manifest: EncoderfileManifest,
    footer: EncoderfileFooter,
    _data: PhantomData<T>,
}
