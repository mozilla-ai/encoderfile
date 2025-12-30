use super::utils::table_to_vec;
use mlua::prelude::*;
use ndarray::{Array1, ArrayD, Axis};
use ndarray_stats::QuantileExt;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor(pub ArrayD<f32>);

impl Tensor {
    pub fn into_inner(self) -> ArrayD<f32> {
        self.0
    }
}

impl FromLua for Tensor {
    fn from_lua(value: LuaValue, _lua: &Lua) -> Result<Tensor, LuaError> {
        match value {
            LuaValue::Table(tbl) => {
                let mut shape = Vec::new();

                let vec = table_to_vec(&tbl, &mut shape)?;

                ArrayD::from_shape_vec(shape.as_slice(), vec)
                    .map_err(|e| {
                        LuaError::external(format!("Failed to cast to dimensionality: {e}"))
                    })
                    .map(Self)
                    .map_err(|e| LuaError::external(format!("Shape error: {e}")))
            }
            LuaValue::UserData(data) => data.borrow::<Tensor>().map(|i| i.to_owned()),
            _ => Err(LuaError::external(
                format!("Unknown type: {}", value.type_name()).as_str(),
            )),
        }
    }
}

impl LuaUserData for Tensor {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        // syntactic sugar
        methods.add_meta_method(LuaMetaMethod::Eq, |_, this, other: Tensor| {
            Ok(this.0 == other.0)
        });

        methods.add_meta_method(LuaMetaMethod::Len, |_, this, _: ()| Ok(this.len()));

        methods.add_meta_method(LuaMetaMethod::Add, |_, this, other| add(this, other));
        methods.add_meta_method(LuaMetaMethod::Sub, |_, this, other| sub(this, other));
        methods.add_meta_method(LuaMetaMethod::Mul, |_, this, other| mul(this, other));
        methods.add_meta_method(LuaMetaMethod::Div, |_, this, other| div(this, other));

        methods.add_meta_method(LuaMetaMethod::ToString, |_, this, _: ()| {
            Ok(this.0.to_string())
        });

        // tensor ops
        methods.add_method("std", |_, this, ddof| this.std(ddof));
        methods.add_method("mean", |_, this, _: ()| this.mean());
        methods.add_method("ndim", |_, this, _: ()| this.ndim());
        methods.add_method("softmax", |_, this, axis: isize| this.softmax(axis));
        methods.add_method("transpose", |_, this, _: ()| this.transpose());
        methods.add_method("lp_normalize", |_, this, (p, axis)| {
            this.lp_normalize(p, axis)
        });
        methods.add_method("min", |_, this, _: ()| this.min());
        methods.add_method("max", |_, this, _: ()| this.max());
        methods.add_method("exp", |_, this, _: ()| this.exp());
        methods.add_method("sum_axis", |_, this, axis| this.sum_axis(axis));
        methods.add_method("sum", |_, this, _: ()| this.sum());

        methods.add_method("map_axis", |_, this, (axis, func)| {
            this.map_axis(axis, func)
        });
        methods.add_method("fold_axis", |_, this, (axis, acc, func)| {
            this.fold_axis(axis, acc, func)
        });
        methods.add_method("mean_pool", |_, this, mask| this.mean_pool(mask));
        methods.add_method("clamp", |_, this, (min, max)| this.clamp(min, max));
        methods.add_method("layer_norm", |_, this, (axis, eps)| {
            this.layer_norm(axis, eps)
        });
        methods.add_method("truncate_axis", |_, this, (axis, len)| {
            this.truncate_axis(axis, len)
        });
    }
}

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

    #[tracing::instrument(skip_all)]
    pub fn truncate_axis(&self, axis: isize, len: usize) -> Result<Self, LuaError> {
        let axis = self.axis1(axis)?;

        let actual_len = self.0.len_of(axis).min(len);

        let mut slice_spec = Vec::with_capacity(self.0.ndim());

        for i in 0..self.0.ndim() {
            if Axis(i) == axis {
                slice_spec.push(ndarray::SliceInfoElem::Slice {
                    start: 0,
                    end: Some(actual_len as isize),
                    step: 1,
                });
            } else {
                slice_spec.push(ndarray::SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                });
            }
        }

        Ok(Tensor(self.0.slice(&slice_spec[..]).to_owned()))
    }

    #[tracing::instrument(skip_all)]
    pub fn clamp(&self, min: Option<f32>, max: Option<f32>) -> Result<Self, LuaError> {
        let input = self
            .0
            .as_slice()
            .ok_or_else(|| LuaError::external("Array must be contiguous"))?;

        let mut out = ArrayD::<f32>::zeros(self.0.raw_dim());
        let out_slice = out
            .as_slice_mut()
            .ok_or_else(|| LuaError::external("Failed to fetch output slice"))?;

        // NaN bound policy: if any bound is NaN, everything becomes NaN. For IEEE-754 compliance :d
        if min.is_some_and(f32::is_nan) || max.is_some_and(f32::is_nan) {
            for dst in out_slice.iter_mut() {
                *dst = f32::NAN;
            }
            return Ok(Self(out));
        }

        match (min, max) {
            (Some(lo), Some(hi)) => {
                for (dst, &src) in out_slice.iter_mut().zip(input.iter()) {
                    *dst = src.max(lo).min(hi);
                }
            }
            (Some(lo), None) => {
                for (dst, &src) in out_slice.iter_mut().zip(input.iter()) {
                    *dst = src.max(lo);
                }
            }
            (None, Some(hi)) => {
                for (dst, &src) in out_slice.iter_mut().zip(input.iter()) {
                    *dst = src.min(hi);
                }
            }
            (None, None) => {
                out_slice.copy_from_slice(input);
            }
        }

        Ok(Self(out))
    }

    #[tracing::instrument(skip_all)]
    pub fn mean_pool(&self, Tensor(mask): Tensor) -> Result<Self, LuaError> {
        assert_eq!(self.0.ndim(), mask.ndim() + 1);

        let ndim = self.0.ndim();

        // Expand mask by adding the last axis back
        let mut mask_expanded = mask.clone();
        mask_expanded = mask_expanded.insert_axis(Axis(ndim - 1));

        // Broadcast mask to full data shape
        let mask_broadcast = mask_expanded
            .broadcast(self.0.shape())
            .ok_or(LuaError::external(format!(
                "cannot broadcast shape {:?} to {:?}",
                mask_expanded.shape(),
                self.0.shape()
            )))?;

        // Multiply and sum over sequence dims (axes 1..ndim-1)
        let weighted = &self.0 * &mask_broadcast;

        // All axes except the last one and the batch axis
        let mut axes_to_reduce = Vec::new();
        for ax in 1..(ndim - 1) {
            axes_to_reduce.push(ax);
        }

        // Sum weighted values
        let mut sum = weighted.clone();
        for ax in axes_to_reduce.iter().rev() {
            sum = sum.sum_axis(Axis(*ax));
        }

        // Sum mask the same way -> counts
        let mut count = mask_expanded.clone();
        for ax in axes_to_reduce.iter().rev() {
            count = count.sum_axis(Axis(*ax));
        }

        // Final: divide elementwise
        Ok(Self(&sum / &count))
    }

    #[tracing::instrument(skip_all)]
    fn fold_axis(&self, axis: isize, acc: f32, func: LuaFunction) -> Result<Tensor, LuaError> {
        let axis = self.axis1(axis)?;

        let mut out = Vec::new();

        for subview in self.0.axis_iter(axis) {
            let mut acc = acc;

            for &x in subview.iter() {
                acc = func.call((acc, x)).map_err(LuaError::external)?;
            }

            out.push(acc);
        }

        let result = Array1::from_shape_vec(out.len(), out)
            .expect("Failed to recast results")
            .into_dyn();

        Ok(Tensor(result))
    }

    #[tracing::instrument]
    fn map_axis(&self, axis: isize, func: LuaFunction) -> Result<Self, LuaError> {
        let axis = self.axis1(axis)?;

        // Pre-size by number of subviews, NOT tensor length.
        let out_len = self.0.shape()[axis.0];
        let mut out = Vec::with_capacity(out_len);

        for subview in self.0.axis_iter(axis) {
            // Only ONE allocation: convert subview into Tensor for Lua
            let tensor_arg = Tensor(subview.to_owned().into_dyn());
            let mapped: Tensor = func.call(tensor_arg).map_err(LuaError::external)?;
            out.push(mapped.0); // store raw ArrayD, not Tensor
        }

        // Stack views without re-wrapping as Tensor
        let views: Vec<_> = out.iter().map(|a| a.view()).collect();

        let stacked = ndarray::stack(axis, &views)
            .map_err(|e| LuaError::external(format!("stack error: {e}")))?;

        Ok(Tensor(stacked))
    }

    #[tracing::instrument(skip_all)]
    fn sum(&self) -> Result<f32, LuaError> {
        Ok(self.0.sum())
    }

    #[tracing::instrument(skip_all)]
    fn sum_axis(&self, axis: isize) -> Result<Self, LuaError> {
        Ok(Self(self.0.sum_axis(self.axis1(axis)?)))
    }

    #[tracing::instrument(skip_all)]
    fn min(&self) -> Result<f32, LuaError> {
        self.0
            .min()
            .copied()
            .map_err(|e| LuaError::external(format!("Min max error: {e}")))
    }

    #[tracing::instrument(skip_all)]
    fn max(&self) -> Result<f32, LuaError> {
        self.0
            .max()
            .copied()
            .map_err(|e| LuaError::external(format!("Min max error: {e}")))
    }

    #[tracing::instrument(skip_all)]
    fn exp(&self) -> Result<Self, LuaError> {
        Ok(Self(self.0.exp()))
    }

    #[tracing::instrument(skip_all)]
    fn lp_normalize(&self, p: f32, axis: isize) -> Result<Self, LuaError> {
        if self.0.is_empty() {
            return Err(LuaError::external("Cannot normalize an empty tensor"));
        }
        if p == 0.0 {
            return Err(LuaError::external("p cannot equal 0.0"));
        }

        let axis = self.axis1(axis)?;
        let arr = &self.0;

        // Compute Lp norm along axis
        let norms = arr.map_axis(axis, |subview| {
            subview
                .iter()
                .map(|&v| v.abs().powf(p))
                .sum::<f32>()
                .powf(1.0 / p)
        });

        // Avoid division by zero using in-place broadcast clamp
        let norms = norms.mapv(|x| if x < 1e-12 { 1e-12 } else { x });

        // Broadcast division using ndarray’s broadcasting API
        let normalized = arr / &norms.insert_axis(axis);

        Ok(Self(normalized))
    }

    fn axis1(&self, axis: isize) -> Result<Axis, LuaError> {
        if axis <= 0 {
            return Err(LuaError::external("Axis must be >= 1."));
        }

        let axis_index = (axis - 1) as usize;

        if axis_index >= self.0.ndim() {
            return Err(LuaError::external("Axis out of range."));
        }

        Ok(Axis(axis_index))
    }

    #[tracing::instrument(skip_all)]
    fn transpose(&self) -> Result<Self, LuaError> {
        Ok(Self(self.0.t().to_owned()))
    }

    #[tracing::instrument(skip_all)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[tracing::instrument(skip_all)]
    fn std(&self, ddof: f32) -> Result<f32, LuaError> {
        Ok(self.0.std(ddof))
    }

    #[tracing::instrument(skip_all)]
    fn mean(&self) -> Result<Option<f32>, LuaError> {
        Ok(self.0.mean())
    }

    #[tracing::instrument(skip_all)]
    fn ndim(&self) -> Result<usize, LuaError> {
        Ok(self.0.ndim())
    }

    #[tracing::instrument(skip_all)]
    fn softmax(&self, axis: isize) -> Result<Self, LuaError> {
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

#[tracing::instrument(skip_all)]
fn add(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            if !is_broadcastable(this.shape(), oth.shape()) {
                return Err(LuaError::external(format!(
                    "Shape {:?} not broadcastable to {:?}",
                    this.shape(),
                    oth.shape()
                )));
            }

            this + oth
        }
        LuaValue::Number(n) => this + (n as f32),
        LuaValue::Integer(i) => this + (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

#[tracing::instrument(skip_all)]
fn sub(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            if !is_broadcastable(oth.shape(), this.shape()) {
                return Err(LuaError::external(format!(
                    "Shape {:?} not broadcastable to {:?}",
                    this.shape(),
                    oth.shape()
                )));
            }

            this - oth
        }
        LuaValue::Number(n) => this - (n as f32),
        LuaValue::Integer(i) => this - (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

#[tracing::instrument(skip_all)]
fn mul(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            if !is_broadcastable(this.shape(), oth.shape()) {
                return Err(LuaError::external(format!(
                    "Shape {:?} not broadcastable to {:?}",
                    this.shape(),
                    oth.shape()
                )));
            }

            this * oth
        }
        LuaValue::Number(n) => this * (n as f32),
        LuaValue::Integer(i) => this * (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

#[tracing::instrument(skip_all)]
fn div(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
    let new = match other {
        LuaValue::UserData(user_data) => {
            let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

            if !is_broadcastable(oth.shape(), this.shape()) {
                return Err(LuaError::external(format!(
                    "Shape {:?} not broadcastable to {:?}",
                    this.shape(),
                    oth.shape()
                )));
            }

            this / oth
        }
        LuaValue::Number(n) => this / (n as f32),
        LuaValue::Integer(i) => this / (i as f32),
        _ => return Err(LuaError::external("Expected either number or Tensor.")),
    };

    Ok(Tensor(new))
}

#[tracing::instrument(skip_all)]
fn is_broadcastable(a: &[usize], b: &[usize]) -> bool {
    let ndim = a.len().max(b.len());

    for i in 0..ndim {
        let ad = *a.get(a.len().wrapping_sub(i + 1)).unwrap_or(&1);
        let bd = *b.get(b.len().wrapping_sub(i + 1)).unwrap_or(&1);

        if ad != bd && ad != 1 && bd != 1 {
            return false;
        }
    }
    true
}
