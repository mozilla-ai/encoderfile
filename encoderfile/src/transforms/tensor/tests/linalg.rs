use super::Tensor;
use ndarray::{ArrayD, Axis};
use ort::tensor::ArrayExtensions;

#[test]
fn test_transpose() {
    let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0], [4.0, 5.0],].into_dyn();
    let transpose = arr.t().into_owned();

    assert_eq!(Tensor(arr).transpose().unwrap().0, transpose)
}

#[test]
fn test_softmax() {
    let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0], [4.0, 5.0],].into_dyn();

    let ts = Tensor(arr.clone());

    assert_eq!(arr.softmax(Axis(0)), ts.softmax(1).unwrap().0);
    assert_eq!(arr.softmax(Axis(1)), ts.softmax(2).unwrap().0);

    assert_ne!(arr.softmax(Axis(1)), ts.softmax(1).unwrap().0);
}

#[test]
fn test_softmax_fail() {
    let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0], [4.0, 5.0],].into_dyn();

    let ts = Tensor(arr.clone());

    assert!(ts.softmax(-1).is_err());
    assert!(ts.softmax(0).is_err());
    assert!(ts.softmax(3).is_err());
}
