use ort::session::Session;
use parking_lot::{Mutex, MutexGuard};
use std::sync::{Arc, OnceLock};

use crate::assets::MODEL_WEIGHTS;

static MODEL: OnceLock<Arc<Mutex<Session>>> = OnceLock::new();

pub type Model<'a> = MutexGuard<'a, Session>;

pub fn get_model() -> Arc<Mutex<Session>> {
    let model = MODEL.get_or_init(|| {
        match Session::builder().and_then(|s| s.commit_from_memory(&MODEL_WEIGHTS)) {
            Ok(model) => Arc::new(Mutex::new(model)),
            Err(e) => panic!("FATAL: Failed to load model: {e:?}"),
        }
    });

    model.clone()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_get_model<'a>() {
        super::get_model();
        assert!(super::MODEL.get().is_some(), "Model not initialized");
    }
}
