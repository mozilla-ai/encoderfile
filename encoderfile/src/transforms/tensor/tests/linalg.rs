use super::{Tensor, load_env};
use ndarray::{ArrayD, Axis};
use ort::tensor::ArrayExtensions;
use mlua::prelude::*;

#[test]
fn test_axis_map() {
    let vec: ArrayD<f32> = ndarray::array![
        [1.0, 2.0],
        [3.0, 4.0]
    ].into_dyn();

    let array_gold = vec
        .axis_iter(Axis(0))
        .map(|i| {
            i.to_owned() / i.mean().unwrap()
        })
        .collect::<Vec<ArrayD<f32>>>();

    let array_gold_refs = array_gold
        .iter()
        .map(|i| i.view())
        .collect::<Vec<_>>();
    
    let Tensor(array_test) = load_env()
        .load("return function(x) return x / x:mean() end")
        .eval::<LuaFunction>()
        .and_then(|func| {
            let tensor = Tensor(vec);
            tensor.axis_map(1, &func)
        })
        .expect("Failed to return test array");

        assert_eq!(
            ndarray::stack(Axis(0), array_gold_refs.as_slice()).unwrap(),
            array_test
        );
}

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
