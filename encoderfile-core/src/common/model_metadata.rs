use super::model_type::ModelTypeEnum;
use std::collections::HashMap;

#[derive(Debug, serde::Serialize, utoipa::ToSchema, utoipa::ToResponse)]
pub struct GetModelMetadataResponse {
    pub model_id: String,
    pub model_type: ModelTypeEnum,
    pub id2label: Option<HashMap<u32, String>>,
}
