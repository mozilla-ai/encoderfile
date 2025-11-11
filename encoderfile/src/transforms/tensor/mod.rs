use super::utils::table_to_vec;
use mlua::prelude::*;
use ndarray::{ArrayD, Axis};
use ndarray_stats::QuantileExt;
use ort::tensor::ArrayExtensions;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor(pub ArrayD<f32>);

impl FromLua for Tensor {
    fn from_lua(value: LuaValue, _lua: &Lua) -> Result<Self, LuaError> {
        match value {
            LuaValue::Table(tbl) => {
                let mut shape = Vec::new();

                let vec = table_to_vec(&tbl, &mut shape)?;

                ArrayD::from_shape_vec(shape.as_slice(), vec)
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
    }
}

impl Tensor {
    fn sum(&self) -> Result<f32, LuaError> {
        Ok(self.0.sum())
    }

    fn sum_axis(&self, axis: isize) -> Result<Self, LuaError> {
        Ok(Self(self.0.sum_axis(self.axis1(axis)?)))
    }

    fn min(&self) -> Result<f32, LuaError> {
        self.0
            .min()
            .map(|i| *i)
            .map_err(|e| LuaError::external(format!("Min max error: {e}")))
    }

    fn max(&self) -> Result<f32, LuaError> {
        self.0
            .max()
            .map(|i| *i)
            .map_err(|e| LuaError::external(format!("Min max error: {e}")))
    }

    fn exp(&self) -> Result<Self, LuaError> {
        Ok(Self(self.0.exp()))
    }

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
            subview.iter().map(|&v| v.abs().powf(p)).sum::<f32>().powf(1.0 / p)
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

    fn transpose(&self) -> Result<Self, LuaError> {
        Ok(Self(self.0.t().to_owned()))
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn std(&self, ddof: f32) -> Result<f32, LuaError> {
        Ok(self.0.std(ddof))
    }

    fn mean(&self) -> Result<Option<f32>, LuaError> {
        Ok(self.0.mean())
    }

    fn ndim(&self) -> Result<usize, LuaError> {
        Ok(self.0.ndim())
    }

    fn softmax(&self, axis: isize) -> Result<Self, LuaError> {
        self.axis1(axis).map(|i| self.0.softmax(i)).map(Self)
    }
}

macro_rules! ops_fn {
    ($fn_name:ident, $op:tt) => {
        fn $fn_name(Tensor(this): &Tensor, other: LuaValue) -> Result<Tensor, LuaError> {
            let new = match other {
                LuaValue::UserData(user_data) => {
                    let Tensor(oth) = user_data.borrow::<Tensor>()?.to_owned();

                    this $op oth
                }
                LuaValue::Number(n) => this $op (n as f32),
                LuaValue::Integer(i) => this $op (i as f32),
                _ => return Err(LuaError::external("Expected either number or Tensor.")),
            };

            Ok(Tensor(new))
        }
    }
}

ops_fn!(add, +);
ops_fn!(sub, -);
ops_fn!(mul, *);
ops_fn!(div, /);
