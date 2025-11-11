use super::Tensor;
use ndarray::{ArrayD, Axis};
use ort::tensor::ArrayExtensions;

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

#[test]
fn test_l1_norm() {
    // L1 norm = sum of absolute values
    let arr: ArrayD<f32> =
        ndarray::array![[1.0, -2.0, 3.0], [4.0, -5.0, 6.0], [7.0, 8.0, -9.0]].into_dyn();

    let tensor = Tensor(arr);
    let norm = tensor.lp_norm(1.0).unwrap();

    let expected = (1..=9).map(|v| v as f32).sum::<f32>();
    assert!(
        (norm - expected).abs() < 1e-6,
        "Expected {expected}, got {norm}"
    );
}

#[test]
fn test_l2_norm() {
    // L2 norm = sqrt(sum(x^2))
    let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0, 2.0]].into_dyn();

    let tensor = Tensor(arr);
    let norm = tensor.lp_norm(2.0).unwrap();

    let expected = (1.0f32.powi(2) + 2.0f32.powi(2) + 2.0f32.powi(2)).sqrt();
    assert!(
        (norm - expected).abs() < 1e-6,
        "Expected {expected}, got {norm}"
    );
}

#[test]
fn test_lp_norm_fractional_p() {
    // fractional p still valid (e.g. p = 0.5)
    let arr: ArrayD<f32> = ndarray::array![[1.0, 4.0]].into_dyn();
    let tensor = Tensor(arr);
    let norm = tensor.lp_norm(0.5).unwrap();

    let expected = (1.0f32.abs().powf(0.5) + 4.0f32.abs().powf(0.5)).powf(1.0 / 0.5);
    assert!(
        (norm - expected).abs() < 1e-6,
        "Expected {expected}, got {norm}"
    );
}

#[test]
fn test_lp_norm_zero_p_fails() {
    // p == 0.0 should error
    let arr: ArrayD<f32> = ndarray::array![[1.0, 2.0, 3.0]].into_dyn();
    let tensor = Tensor(arr);
    let result = tensor.lp_norm(0.0);
    assert!(result.is_err());
}

#[test]
fn test_lp_norm_empty_tensor() {
    // empty tensors should return 0 (sum over empty iterator)
    let arr: ArrayD<f32> = ArrayD::<f32>::zeros(ndarray::IxDyn(&[0]));
    let tensor = Tensor(arr);
    let norm = tensor.lp_norm(2.0).unwrap();
    assert_eq!(norm, 0.0);
}

#[test]
fn test_linf_norm() {
    // L∞ norm = max(|x_i|)
    let arr: ArrayD<f32> = ndarray::array![[1.0, -4.0, 2.0], [3.0, 9.0, -7.0]].into_dyn();

    let tensor = Tensor(arr);
    let norm = tensor.lp_norm(f32::INFINITY).unwrap();

    let expected = 9.0;
    assert!(
        (norm - expected).abs() < 1e-6,
        "Expected {expected}, got {norm}"
    );
}
