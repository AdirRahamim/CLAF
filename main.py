import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import argparse
from torchvision import datasets, transforms
from models.resnet import ResNet18 as resnet18
from utils import test_model, adv_test_model, TwoCropsTransform, SupConLoss, adjust_learning_rate
import os
import torchattacks


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.fc(x)


class Model(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(Model, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = self.classifier(features)
        return outputs

    def forward_features(self, x):
        features = self.feature_extractor(x)
        return features


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for model training')

    parser.add_argument('--save-path', type=str, default='./Saved_models',
                        help='Path to save checkpoints')
    parser.add_argument('--lr', type=int, default=0.05, help='lr for optimizer')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200,
                        help='Total number of epochs')
    parser.add_argument('--aug-epochs', type=int, default=60,
                        help='Number of augmentation_only_epochs')
    parser.add_argument('--linear-classifier-epochs', type=int, default=5,
                        help='Num of epochs for linear classifier training')
    parser.add_argument('--linear-classifier-pgd-steps', type=int, default=5)

    # Ablation study
    parser.add_argument('--same-classifier', action='store_true', default=True,
                        help='Use same classifier or reinitialize at each encoder epoch')
    parser.add_argument('--adv-classifier', action='store_true', default=True,
                        help='Adversarial train linear classifier')

    parser.add_argument('--lr-c', type=int, default=1e-3,
                        help='lr for linear classifier train')
    parser.add_argument('--epochs-c', type=int, default=100,
                        help='Num of epochs to train on linear classifier train')
    args = parser.parse_args()
    return args


def pre_train_aug(model: Model, projection_head, loader, optimizer, device, criterion):
    model.train()
    train_loss = 0
    for x, y in loader:
        _, x1, x2 = x
        images, y = torch.cat([x1, x2], dim=0).to(device), y.to(device)
        bsz = y.shape[0]
        features = projection_head(F.normalize(model.forward_features(images), dim=1))
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    loss = train_loss / len(loader)
    print(f'[CL AUG] Loss: {loss:.3f}')
    return loss


def pre_train_adv(model: Model, projection_head, in_dim, loader_augment, loader_classifier, optimizer,
                  device, criterion, args):
    # Pretrain using augmentations and adversarial images
    for param_f, param_p in zip(model.feature_extractor.parameters(), projection_head.parameters()):
        param_f.requires_grad = False
        param_p.requires_grad = False

    if not args.same_classifier:
        model.classifier = LinearClassifier(in_dim, 10).to(device)
    model.train()
    optimizer_c = Adam(model.classifier.parameters(), lr=1e-3)

    attack = torchattacks.PGD(model, eps=8/255, steps=args.linear_classifier_pgd_steps)
    train_loss, num_correct, total = 0, 0, 0
    for i in range(args.linear_classifier_epochs):
        train_loss, num_correct, total = 0, 0, 0
        for x, y in loader_classifier:
            x, y = x.to(device), y.to(device)
            optimizer_c.zero_grad()
            if args.adv_classifier:
                adv_x = attack(x, y)
                outputs = model(adv_x)
            else:
                outputs = model(x)
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            optimizer_c.step()
            pred = outputs.argmax(dim=1)
            num_correct += torch.sum(pred.eq(y)).item()
            total += y.numel()
            train_loss += loss.item()
    print(f'[TRAIN Classifier] Acc: {100. * num_correct / total:.3f}%')

    for param_f, param_p in zip(model.feature_extractor.parameters(), projection_head.parameters()):
        param_f.requires_grad = True
        param_p.requires_grad = True

    train_loss = 0
    for x_aug, y in loader_augment:
        x, x1, x2 = x_aug
        x, x1, x2, y = x.to(device), x1.to(device), x2.to(device), y.to(device)
        x_adv = attack(x, y)

        images = torch.cat([x1, x2, x_adv], dim=0).to(device)
        bsz = y.shape[0]
        features = projection_head(F.normalize(model.forward_features(images), dim=1))

        f1, f2, f_adv = torch.split(features, [bsz, bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1), f_adv.unsqueeze(1)], dim=1)
        loss = criterion(features, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    loss = train_loss / len(loader_augment)
    print(f'[CL ADV] Loss: {loss:.3f}')
    return loss


def save_model(model, epoch, optimizer, args, name=None):
    if name is None:
        model_out = os.path.join(args.save_path, args.defense_type)
    else:
        model_out = os.path.join(args.save_path, name)
    state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, model_out + '.pth')


def contrastive_train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'-> device: {device}')

    pretrain_aug_transform = TwoCropsTransform(transforms.Compose([
                            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
                    transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

    dataset1 = datasets.CIFAR10('data/cifar10', train=True, transform=pretrain_aug_transform, download=True)
    pretrain_aug_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, batch_size=args.batch_size,
                                                      pin_memory=True,
                                                      num_workers=2)

    feature_extractor = resnet18(num_classes=128, is_contrastive=True)
    in_dim, out_dim = 512, 128
    projection_head = ProjectionHead(in_dim, out_dim).to(device)
    model = Model(feature_extractor, None).to(device)
    optimizer = torch.optim.Adam(list(model.feature_extractor.parameters()) + list(projection_head.parameters()),
                                 args.lr, weight_decay=args.weight_decay)
    criterion = SupConLoss(temperature=0.07).to(device)

    start_epoch = 0
    if os.path.isfile(os.path.join(args.save_path, 'model_aug') + '.pth'):
        checkpoint = torch.load(os.path.join(args.save_path, 'model_aug') + '.pth',
                                map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        checkpoint = torch.load(os.path.join(args.save_path, 'Projection_head') + '.pth', map_location=device)
        projection_head.load_state_dict(checkpoint['model_state'])

        print(f'Loaded checkpoint at epoch {start_epoch}')

    # Feature extractor pretrain using augmentation only
    for epoch in range(start_epoch, args.aug_epochs):
        print(f'Epoch pretrain [AUGMENT] {epoch}')
        loss = pre_train_aug(model, projection_head, pretrain_aug_loader, optimizer, device, criterion)
        adjust_learning_rate(args.lr, args.lr_decay_rate, epoch, args.epochs, optimizer)
        save_model(model, epoch, optimizer, args, name='model_aug')
        save_model(projection_head, epoch, optimizer, args, name='Projection_head')

    transform_classifier = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    dataset3 = datasets.CIFAR10('data/cifar10', train=True, transform=transform_classifier, download=True)
    classifier_loader = torch.utils.data.DataLoader(dataset3, shuffle=True, batch_size=args.batch_size,
                                                    pin_memory=True, num_workers=2)

    start2 = args.aug_epochs
    model.classifier = LinearClassifier(in_dim, 10).to(device)
    if os.path.isfile(os.path.join(args.save_path, 'model') + '.pth'):
        checkpoint = torch.load(os.path.join(args.save_path, 'model') + '.pth',
                                map_location=device)
        start2 = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = torch.load(os.path.join(args.save_path, 'Projection_head_adv') + '.pth',
                                map_location=device)
        projection_head.load_state_dict(checkpoint['model_state'])
        print(f'Loaded checkpoint at epoch {start2}')

    for epoch in range(start2, args.epochs, 1):
        print(f'Epoch pretrain [ADV] {epoch}')
        pre_train_adv(model, projection_head, in_dim, pretrain_aug_loader, classifier_loader,
                             optimizer, device, criterion, args)
        adjust_learning_rate(args.lr, args.lr_decay_rate, epoch, args.epochs, optimizer)
        save_model(model, epoch, optimizer, args, name='model')
        save_model(projection_head, epoch, optimizer, args, name='Projection_head_adv')

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    dataset4 = datasets.CIFAR10('data/cifar10', train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset4, shuffle=False, batch_size=args.batch_size,
                                              pin_memory=True, num_workers=2)

    model.classifier = LinearClassifier(in_dim, 10).to(device)
    model.train()
    optimizer_c = Adam(model.classifier.parameters(), lr=args.lr_c)
    train_loss, num_correct, total = 0, 0, 0
    for epoch in range(args.epochs_c):
        print(f'Epoch classifier {epoch}')
        train_loss, num_correct, total = 0, 0, 0
        for x, y in classifier_loader:
            x, y = x.to(device), y.to(device)
            optimizer_c.zero_grad()
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            optimizer_c.step()
            pred = outputs.argmax(dim=1)
            num_correct += torch.sum(pred.eq(y)).item()
            total += y.numel()
            train_loss += loss.item()
        print(f'[TRAIN Classifier] Acc: {100. * num_correct / total:.3f}%')
        adjust_learning_rate(args.lr_c, 0.2, epoch, args.epochs_c, optimizer)
        acc = test_model(model, test_loader, device)
        print(f'[TEST] Acc: {acc * 100:.3f}%')
        save_model(model, epoch, optimizer, args, name='model_classifier')

    print(f'[TRAIN Classifier] Acc: {100. * num_correct / total:.3f}%')
    attack = torchattacks.PGD(model, eps=8/255, steps=10)
    num_correct, total = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        adv_x = attack(x, y)
        outputs = model(adv_x)
        pred = outputs.argmax(dim=1)
        num_correct += torch.sum(pred.eq(y)).item()
        total += y.numel()
    print(f'Adv accuracy under PGD-10 eps: 8/255: {num_correct / total * 100:.3f}%')

    attack = torchattacks.PGD(model, eps=16/255, steps=10)
    num_correct, total = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        adv_x = attack(x, y)
        outputs = model(adv_x)
        pred = outputs.argmax(dim=1)
        num_correct += torch.sum(pred.eq(y)).item()
        total += y.numel()
    print(f'Adv accuracy under PGD-10 eps: 16/255: {num_correct / total * 100:.3f}%')


if __name__ == '__main__':
    args = get_args()
    print(args)
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    contrastive_train(args)
