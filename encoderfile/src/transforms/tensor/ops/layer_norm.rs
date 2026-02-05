use mlua::prelude::*;
use super::Tensor;

impl Tensor {
    #[tracing::instrument(skip_all)]
    pub fn layer_norm(&self, axis: isize, eps: f32) -> Result<Self, LuaError> {
        // normalize over axis
        let axis = self.axis1(axis)?;
        let mean = self
            .0
            .mean_axis(axis)
            .ok_or(LuaError::external(
                "Failed to mean_axis Tensor: Axis length must be > 0.",
            ))?
            .insert_axis(axis);

        // no bias: ddof = 0.0
        let var = self.0.var_axis(axis, 0.0);
        let std = (var + eps).mapv(f32::sqrt).insert_axis(axis);

        // y = (x − mean(x)) / sqrt(var(x) + eps)
        Ok(Tensor(((&self.0 - &mean) / &std).into_dyn()))
    }
}

#[test]
fn test_layer_norm_correctness() {
    let input = Tensor(ndarray::array![[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]].into_dyn());

    let result = input
        .layer_norm(2, 1e-5)
        .expect("Failed to compute layer_norm");

    for row in result.0.rows() {
        let m = row.mean().unwrap();
        let v = row.var(0.0);
        // mean should be roughly equal to 0
        assert!((m - 0.0).abs() < 1e-5);
        // variance tolerances are always a bit looser, but should roughly equal 1.0
        assert!((v - 1.0).abs() < 1e-4);
    }
}

#[test]
fn test_layer_norm_epsilon_behavior() {
    let input = Tensor(ndarray::array![[5.0, 5.0, 5.0]].into_dyn());
    let result = input
        .layer_norm(2, 1e-5)
        .expect("Failed to compute layer_norm");

    // nothing should blow up or be NaNs
    assert!(result.0.iter().all(|v| v.is_finite()));
}

#[test]
fn test_layer_norm_dimensionality() {
    use ndarray::Array3;
    let input = Tensor(Array3::from_elem([10, 10, 10], 3.0).into_dyn());
    let result = input
        .layer_norm(2, 1e-5)
        .expect("Failed to compute layer_norm");
    assert_eq!(input.0.dim(), result.0.dim())
}

#[test]
fn test_layer_norm_translation() {
    use super::arithm::add;

    // layer_norm should be invariant to additive bias per row
    let input_1 = Tensor(ndarray::array![[1.0, 2.0, 3.0, 4.0, 5.0]].into_dyn());
    let input_2 = add(&input_1, LuaValue::Number(5.0)).expect("Scalar transformation failed");
    let layer_norm_1 = input_1
        .layer_norm(2, 1e-5)
        .expect("Failed to compute layer_norm for input_1");
    let layer_norm_2 = input_2
        .layer_norm(2, 1e-5)
        .expect("Failed to compute layer_norm for input_2");

    for (a, b) in layer_norm_1.0.iter().zip(layer_norm_2.0.iter()) {
        assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
    }
}

