use mlua::prelude::*;

pub fn table_to_vec(t: &LuaTable, shape: &mut Vec<usize>) -> Result<Vec<f32>, LuaError> {
    let mut data = Vec::new();

    let mut len = 0;
    for pair in t.clone().pairs::<LuaValue, LuaValue>() {
        let (_, v) = pair?;
        len += 1;
        match v {
            LuaValue::Table(subt) => {
                let mut subshape = Vec::new();
                let subdata = table_to_vec(&subt, &mut subshape)?;
                if shape.is_empty() {
                    shape.extend(std::iter::once(len).chain(subshape));
                } else {
                    shape[0] = len;
                }
                data.extend(subdata);
            }
            LuaValue::Integer(i) => data.push(i as f32),
            LuaValue::Number(n) => data.push(n as f32),
            _ => {
                return Err(LuaError::FromLuaConversionError {
                    from: v.type_name(),
                    to: "f32 or nested table".to_string(),
                    message: Some("Invalid value in tensor table".into()),
                });
            }
        }
    }

    if shape.is_empty() {
        shape.push(len);
    }

    Ok(data)
}
