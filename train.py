import os
import numpy as np
import torch
from tqdm import tqdm
from proto_batch_sampler import PrototypicalBatchSampler
from proto_loss import prototypical_loss as loss_fn
from dataset import TrainDataset
from proto_encoder import PrototypeEncoder

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    """
    Train the model with the prototypical learning algorithm.
    Save the best and final models during training in the current working directory.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    best_acc = 0
    best_state = model.state_dict()  # Initialize with the current state of the model

    # Paths to save models in the current working directory
    cwd = os.getcwd()
    best_model_path = os.path.join(cwd, 'best_model.pth')
    last_model_path = os.path.join(cwd, 'last_model.pth')

    for epoch in range(opt['epochs']):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y, n_support=opt['num_support_tr'])
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt['iterations']:])
        avg_acc = np.mean(train_acc[-opt['iterations']:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()

        if val_dataloader is None:
            continue

        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y, n_support=opt['num_support_val'])
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt['iterations']:])
        avg_acc = np.mean(val_acc[-opt['iterations']:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))

        # Save the best model
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")
            best_acc = avg_acc
            best_state = model.state_dict()

    # Save the final model
    torch.save(model.state_dict(), last_model_path)
    print(f"Final model saved to {last_model_path}")

    # Save training and validation metrics
    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(cwd, name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def main(opt):
    # Seed initialization
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt['manual_seed'])
    torch.manual_seed(opt['manual_seed'])
    torch.cuda.manual_seed(opt['manual_seed'])

    # Dataset and DataLoader
    dataset = TrainDataset(opt['train_dataset_path'])
    n_classes = len(np.unique(dataset.y_train))
    if n_classes < opt['classes_per_it_tr'] or n_classes < opt['classes_per_it_val']:
        raise Exception(
            'There are not enough classes in the dataset to satisfy the chosen classes_per_it. '
            'Decrease the classes_per_it_{tr/val} option and try again.')

    sampler = PrototypicalBatchSampler(
        labels=dataset.y_train,
        classes_per_it=opt['classes_per_it_tr'],
        num_samples=opt['num_support_tr'] + opt['num_query_tr'],
        iterations=opt['iterations']
    )
    tr_dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    # Model, Optimizer, and Scheduler
    device = opt['device']
    model = PrototypeEncoder().to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=opt['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optim,
        step_size=opt['lr_scheduler_step'],
        gamma=opt['lr_scheduler_gamma']
    )

    # Training
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = train(opt, tr_dataloader, model, optim, lr_scheduler)
    model.load_state_dict(best_state)

    return model
