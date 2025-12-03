use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub pad_token_id: u32,
    pub num_labels: Option<usize>,
    pub id2label: Option<HashMap<u32, String>>,
    pub label2id: Option<HashMap<String, u32>>,
}

impl ModelConfig {
    pub fn id2label(&self, id: u32) -> Option<&str> {
        self.id2label.as_ref()?.get(&id).map(|s| s.as_str())
    }

    pub fn label2id(&self, label: &str) -> Option<u32> {
        self.label2id.as_ref()?.get(label).copied()
    }

    pub fn num_labels(&self) -> Option<usize> {
        if self.num_labels.is_some() {
            return self.num_labels;
        }

        if let Some(id2label) = &self.id2label {
            return Some(id2label.len());
        }

        if let Some(label2id) = &self.label2id {
            return Some(label2id.len());
        }

        None
    }
}
