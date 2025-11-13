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

#[cfg(test)]
mod tests {
    use super::*;
    use mlua::Lua;

    fn run_table_to_vec(lua_code: &str) -> (Vec<f32>, Vec<usize>) {
        let lua = Lua::new();
        let t: LuaTable = lua.load(lua_code).eval().unwrap();
        let mut shape = Vec::new();
        let data = table_to_vec(&t, &mut shape).unwrap();
        (data, shape)
    }

    #[test]
    fn test_flat_table() {
        let (data, shape) = run_table_to_vec("{1, 2, 3, 4}");
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, vec![4]);
    }

    #[test]
    fn test_nested_table_2d() {
        let (data, shape) = run_table_to_vec("{{1, 2, 3}, {4, 5, 6}}");
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    fn test_nested_table_3d() {
        let (data, shape) = run_table_to_vec("{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}");
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(shape, vec![2, 2, 2]);
    }

    #[test]
    fn test_mixed_number_types() {
        let (data, shape) = run_table_to_vec("{1, 2.5, 3, 4.5}");
        assert_eq!(data, vec![1.0, 2.5, 3.0, 4.5]);
        assert_eq!(shape, vec![4]);
    }

    #[test]
    fn test_invalid_value() {
        let lua = Lua::new();
        let t: LuaTable = lua.load("{1, 'bad', 3}").eval().unwrap();
        let mut shape = Vec::new();
        let err = table_to_vec(&t, &mut shape).unwrap_err();
        if let LuaError::FromLuaConversionError { message, .. } = err {
            assert!(message.unwrap().contains("Invalid value"));
        } else {
            panic!("Expected FromLuaConversionError");
        }
    }

    #[test]
    fn test_empty_table() {
        let (data, shape) = run_table_to_vec("{}");
        assert_eq!(data, Vec::<f32>::new());
        assert_eq!(shape, vec![0]);
    }
}
