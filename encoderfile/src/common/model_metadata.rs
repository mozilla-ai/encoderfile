use super::model_type::ModelType;
use std::collections::HashMap;

#[derive(Debug, serde::Serialize, utoipa::ToSchema, utoipa::ToResponse)]
pub struct GetModelMetadataResponse {
    pub model_id: String,
    pub model_type: ModelType,
    pub id2label: Option<HashMap<u32, String>>,
}

impl From<GetModelMetadataResponse> for crate::generated::metadata::GetModelMetadataResponse {
    fn from(val: GetModelMetadataResponse) -> Self {
        Self {
            model_id: val.model_id,
            model_type: crate::generated::metadata::ModelType::from(val.model_type).into(),
            id2label: val.id2label.unwrap_or_default(),
        }
    }
}
