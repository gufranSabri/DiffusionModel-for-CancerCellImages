import torch
import torchvision.models as models

def modify_resnet101(model, num_classes):
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model

def modify_vgg16(model, num_classes):
    in_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(in_features, num_classes)
    return model

def modify_mobilenetv2(model, num_classes):
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model

def modify_inceptionv3(model, num_classes):
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model

def modify_densenet121(model, num_classes):
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)
    return model

def get_model(model_name, num_classes=2, pretrained=True):
    model_dict = {
        "MobileNetV2": models.mobilenet_v2,
        "ResNet101": models.resnet101,
        "VGG16": models.vgg16,
        "InceptionV3": models.inception_v3,
        "DenseNet121": models.densenet121,
    }

    if model_name in model_dict:
        model = model_dict[model_name](pretrained=pretrained)
        
        if model_name == "ResNet101":
            model = modify_resnet101(model, num_classes)
        # elif model_name == "VGG16":
        #     model = modify_vgg16(model, num_classes)
        elif model_name == "MobileNetV2":
            model = modify_mobilenetv2(model, num_classes)
        # elif model_name == "InceptionV3":
        #     model = modify_inceptionv3(model, num_classes)
        elif model_name == "DenseNet121":
            model = modify_densenet121(model, num_classes)

        return model
    else:
        raise ValueError(f"Model not defined: {model_name}")