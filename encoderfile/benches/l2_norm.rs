use encoderfile::inference::utils::l2_normalize;
use ndarray::{Array2, Axis};
use divan::Bencher;

fn main() {
    divan::main();
}

#[divan::bench(args = [Axis(0), Axis(1)])]
fn bench_l2_normalize(bencher: Bencher, axis: Axis) {
    // create a representative test array
    let x = Array2::<f32>::from_shape_fn((1024, 256), |(i, j)| (i + j) as f32);

    bencher.bench(|| {
        // clone to avoid mutating the same array
        let arr = x.clone();
        l2_normalize(arr, axis)
    });
}
