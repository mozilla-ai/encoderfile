use crate::transforms::Transform;

#[cfg(not(tarpaulin_include))]
pub fn get_transform(transform_str: Option<&str>) -> Transform {
    if let Some(script) = transform_str {
        let engine = Transform::new(script).expect("Failed to create transform");

        return engine;
    }

    Transform::new("").expect("Failed to create transform")
}
