use ort::session::Session;
use std::sync::OnceLock;
use parking_lot::{Mutex, MutexGuard};

use crate::assets::MODEL_WEIGHTS;

static MODEL: OnceLock<Mutex<Session>> = OnceLock::new();

pub fn get_model<'a>() -> MutexGuard<'static, Session> {
    let model = MODEL.get_or_init(|| {
        match Session::builder().and_then(|s| s.commit_from_memory(MODEL_WEIGHTS)) {
            Ok(model) => Mutex::new(model),
            Err(e) => panic!("FATAL: Failed to load model: {e:?}"),
        }
    });

    model.lock()
}
