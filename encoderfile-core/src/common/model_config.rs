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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_labels() {
        let test_labels: Vec<(String, u32)> = vec![("a", 1), ("b", 2), ("c", 3)]
            .into_iter()
            .map(|(i, j)| (i.to_string(), j))
            .collect();

        let label2id: HashMap<String, u32> = test_labels.clone().into_iter().collect();
        let id2label: HashMap<u32, String> = test_labels
            .clone()
            .into_iter()
            .map(|(i, j)| (j, i))
            .collect();

        let config = ModelConfig {
            model_type: "MyModel".to_string(),
            pad_token_id: 0,
            num_labels: Some(3),
            id2label: Some(id2label.clone()),
            label2id: Some(label2id.clone()),
        };

        assert_eq!(config.num_labels(), Some(3));

        let config = ModelConfig {
            model_type: "MyModel".to_string(),
            pad_token_id: 0,
            num_labels: None,
            id2label: Some(id2label.clone()),
            label2id: Some(label2id.clone()),
        };

        assert_eq!(config.num_labels(), Some(3));

        let config = ModelConfig {
            model_type: "MyModel".to_string(),
            pad_token_id: 0,
            num_labels: None,
            id2label: None,
            label2id: Some(label2id.clone()),
        };

        assert_eq!(config.num_labels(), Some(3));
    }
}
