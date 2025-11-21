use super::Tensor;
use ndarray::{Array3, ArrayD, Axis, array, s};

#[test]
fn test_transpose() {
    let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0], [4.0, 5.0],].into_dyn();
    let transpose = arr.t().into_owned();

    assert_eq!(Tensor(arr).transpose().unwrap().0, transpose)
}

#[test]
fn test_softmax() {
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
    let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();

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
    let arr = array![[-1000.0, -1000.0, -1000.0]].into_dyn();

    let Tensor(sm) = Tensor(arr).softmax(1).unwrap();
    let sum: f32 = sm.sum_axis(Axis(0))[0];

    assert!((sum - 1.0).abs() < 1e-6);
    assert!(!sm.iter().any(|x| x.is_nan()));
}

#[test]
fn test_softmax_peaked_distribution() {
    let arr = array![[0.0, 0.0, 100.0]].into_dyn();

    let Tensor(sm) = Tensor(arr).softmax(1).unwrap();

    assert!(sm[[0, 2]] > 0.999);
}

#[test]
fn test_softmax_fail() {
    let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0], [4.0, 5.0],].into_dyn();

    let ts = Tensor(arr.clone());

    assert!(ts.softmax(-1).is_err());
    assert!(ts.softmax(0).is_err());
    assert!(ts.softmax(3).is_err());
}

#[test]
fn test_lp_norm_empty() {
    let arr: ArrayD<f32> = ndarray::array![[[]]].into_dyn();

    assert!(arr.is_empty());
    assert!(Tensor(arr).lp_normalize(1.0, 1).is_err())
}

#[test]
fn test_lp_norm_zero() {
    let arr: ArrayD<f32> = Array3::ones((3, 3, 3)).into_dyn();

    assert!(Tensor(arr).lp_normalize(0.0, 1).is_err())
}

#[test]
fn test_lp_norm_nonexistent_dim() {
    let arr: ArrayD<f32> = Array3::ones((3, 3, 3)).into_dyn();

    assert!(Tensor(arr.clone()).lp_normalize(1.0, 0).is_err()); // lua starts with 1
    assert!(Tensor(arr.clone()).lp_normalize(1.0, 4).is_err());
}
