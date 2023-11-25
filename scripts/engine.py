import torch.nn
import torch
import torchmetrics
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger('Model Training')
logger.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')

logger = logging.getLogger('Classification')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Normalization(input_data):

    mean = torch.mean(input_data, 2)
    std = torch.std(input_data, 2)

    return (input_data-mean.unsqueeze(2)) / std.unsqueeze(2)


def train(epoch, dataloader: torch.utils.data.DataLoader,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
          device: str
          ):
    model.train()
    loss_total = AverageMeter()
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2).cuda()

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs.half())

        loss = loss_fn(outputs.half(), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total.update(loss)
        accuracy.update(torch.argmax(outputs, dim=1), targets)

    acc = accuracy.compute()
    # writer.add_scalar('Loss/train', loss_total.avg.item(), epoch)
    # writer.add_scalar('Acc/train', acc.item(), epoch)
    logger.info(f'Train: Epoch:{epoch} Loss:{loss_total.avg:.4} Accuracy:{acc:.4}')
    return loss_total.avg.item(), acc.item()


def valid(epoch, dataloader: torch.utils.data.DataLoader,
          model: torch.nn.Module, loss_fn: torch.nn.Module,
          device: str,
          checkpoint):
    model.eval()
    loss_total = AverageMeter()
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.half())

            loss = loss_fn(outputs.half(), targets)
            loss_total.update(loss)
            accuracy.update(torch.argmax(outputs, dim=1), targets)

        acc = accuracy.compute()
        logger.info(f'validation: Epoch:{epoch} Loss:{loss_total.avg:.4} Accuracy:{acc:.4}')
    # Save checkpoint
    checkpoint.save(accuracy.compute(), epoch=epoch)
    return loss_total.avg.item(), acc.item()


def result(EXPERIMENTS_SAVE_ROOT, EXPERIMENT_ID, train_loss, valid_loss, train_accuracy, valid_accuracy,
           data_loader_test, model, device):
    reports_dir = EXPERIMENTS_SAVE_ROOT / EXPERIMENT_ID / 'reports'
    if not reports_dir.exists():
        reports_dir.mkdir(parents=True)

    plt.figure(figsize=(15, 15))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'valid'], loc='upper right')

    plt.savefig(reports_dir / r"epoch_loss.png")
    plt.close()

    plt.figure(figsize=(15, 15))
    plt.plot(train_accuracy)
    plt.plot(valid_accuracy)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['training', 'valid'], loc='upper right')

    plt.savefig(reports_dir / r"epoch_Accuracy.png")
    plt.close()

    truelabels = []
    predictions = []

    model.eval()
    for batch_idx, (inputs, targets) in enumerate(data_loader_test):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs.half())

        for label in targets:
            truelabels.append(label.item())

        for prediction in torch.argmax(outputs, dim=1):
            predictions.append(prediction.item())

    # Plot the confusion matrix
    cm = confusion_matrix(truelabels, predictions)
    df_cm = pd.DataFrame(cm, index=['NON_OK', 'OK'],
                         columns=['NON_OK', 'OK'])

    plt.figure(figsize=(7, 7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel("Predicted Shape", fontsize=20)
    plt.ylabel("True Shape", fontsize=20)
    plt.savefig(reports_dir / r"confusion_numerical.png")
    plt.close()

    cm = confusion_matrix(truelabels, predictions, normalize='true')
    df_cm = pd.DataFrame(cm, index=['NON_OK', 'OK'],
                         columns=['NON_OK', 'OK'])

    plt.figure(figsize=(7, 7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel("Predicted Shape", fontsize=20)
    plt.ylabel("True Shape", fontsize=20)
    plt.savefig(reports_dir / r"confusion_percent.png")
    plt.close()


if __name__ == '__main__':
    print('This is Main!')
