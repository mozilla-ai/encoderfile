use crate::common;

tonic::include_proto!("encoderfile.image_types");

impl From<common::ImageInfo> for ImageInput {
    fn from(val: common::ImageInfo) -> Self {
        ImageInput {
            image: val.image_bytes.to_vec(),
        }
    }
}

impl From<common::ImageLabelScore> for ImageLabelScore {
    fn from(val: common::ImageLabelScore) -> Self {
        ImageLabelScore {
            label: val.label,
            score: val.score,
        }
    }
}

impl From<common::ImageLabels> for ImageLabels {
    fn from(val: common::ImageLabels) -> Self {
        ImageLabels {
            labels: val.labels.into_iter().map(|label| label.into()).collect(),
        }
    }
}
