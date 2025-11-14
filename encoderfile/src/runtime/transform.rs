use crate::transforms::Transform;

include!(concat!(env!("OUT_DIR"), "/generated/transform.rs"));

#[cfg(not(tarpaulin_include))]
pub fn get_transform() -> Transform {
    if let Some(script) = TRANSFORM {
        let engine = Transform::new(script).expect("Failed to create transform");

        return engine;
    }

    Transform::new("").expect("Failed to create transform")
}
