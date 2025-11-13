use crate::transforms::Transform;

include!(concat!(
    env!("OUT_DIR"),
    "/generated/transform.rs"
));

pub fn get_transform() -> Option<Transform> {
    if let Some(script) = TRANSFORM {
        let engine = Transform::new(script).expect("Failed to create transform");

        return Some(engine);
    }

    None
}
