from torchvision import transforms

def build_train_transforms(image_size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        transforms.ColorJitter(
            brightness = 0.15,
            contrast = 0.15,
            saturation = 0.15,
            hue = 0.02
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

def build_eval_transforms(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])


