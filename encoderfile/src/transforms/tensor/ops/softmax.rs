use mlua::prelude::*;
use super::Tensor;

impl Tensor {
    #[tracing::instrument(skip_all)]
    pub fn softmax(&self, axis: isize) -> Result<Self, LuaError> {
        let axis = self.axis1(axis)?;

        let max_vals = self.0.map_axis(axis, |row| {
            row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v))
        });

        let z = &self.0 - &max_vals.insert_axis(axis);

        let numerator = z.mapv(|x| x.exp());

        let denom = numerator.map_axis(axis, |row| row.sum());

        Ok(Tensor(numerator / &denom.insert_axis(axis)))
    }
}

#[test]
fn test_softmax() {
    use ndarray::{ArrayD, s};
    let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0],].into_dyn();

    // softmax along second axis
    // remember — function is 1-indexed

    let Tensor(softmaxed) = Tensor(arr).softmax(2).expect("Failed to softmax");

    let arr1 = softmaxed.slice(s![0, ..]);
    let arr2 = softmaxed.slice(s![1, ..]);

    assert_eq!(arr1, arr2);
}

#[test]
fn test_softmax_rows_sum_to_one() {
    use ndarray::Axis;
    let arr = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();

    // iterate over second axis in lua-land, first axis in rust land
    let Tensor(sm) = Tensor(arr).softmax(2).unwrap();

    // should iterate over 0th axis
    for row in sm.axis_iter(Axis(0)) {
        let sum = row.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_softmax_large_negative_values() {
    use ndarray::Axis;
    let arr = ndarray::array![[-1000.0, -1000.0, -1000.0]].into_dyn();

    let Tensor(sm) = Tensor(arr).softmax(1).unwrap();
    let sum: f32 = sm.sum_axis(Axis(0))[0];

    assert!((sum - 1.0).abs() < 1e-6);
    assert!(!sm.iter().any(|x| x.is_nan()));
}

#[test]
fn test_softmax_peaked_distribution() {
        let arr = ndarray::array![[0.0, 0.0, 100.0]].into_dyn();

    let Tensor(sm) = Tensor(arr).softmax(1).unwrap();

    assert!(sm[[0, 2]] > 0.999);
}

#[test]
fn test_softmax_fail() {
    use ndarray::ArrayD;
    let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0], [4.0, 5.0],].into_dyn();

    let ts = Tensor(arr.clone());

    assert!(ts.softmax(-1).is_err());
    assert!(ts.softmax(0).is_err());
    assert!(ts.softmax(3).is_err());
}

