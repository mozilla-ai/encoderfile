use ort::session::Session;
use parking_lot::{Mutex, MutexGuard};
use std::sync::{Arc, OnceLock};

static MODEL: OnceLock<Arc<Mutex<Session>>> = OnceLock::new();

pub fn get_model(model_bytes: &[u8]) -> Arc<Mutex<Session>> {
    let model = MODEL.get_or_init(|| {
        match Session::builder().and_then(|s| s.commit_from_memory(model_bytes)) {
            Ok(model) => Arc::new(Mutex::new(model)),
            Err(e) => panic!("FATAL: Failed to load model: {e:?}"),
        }
    });

    model.clone()
}

pub type Model<'a> = MutexGuard<'a, Session>;
