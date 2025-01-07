from transformers import AutoImageProcessor, ResNetForImageClassification,AutoModel
from torch import nn

model_name = "microsoft/resnet-50"
def create_model(num_classes=3, id2label=None, label2id=None, device="cpu"):
    
    # Load the model with proper configuration
    model = ResNetForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # Required when changing number of classes
    )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model = model.to(device)

    return model