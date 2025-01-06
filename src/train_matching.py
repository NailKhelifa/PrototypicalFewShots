import torch
import torch.backends.cudnn as cudnn
import tqdm
from torch.autograd import Variable
from src.test_model import load_data, get_n_per_classes
import os 
from src.test_model import load_data, get_n_per_classes
from src.dataset import SignalNShotTrainDataset
from src.matching import MatchingNetwork


def run_training_epoch(data, opt, device, total_train_batches, optim, model, lr_scheduler):
    """
    Runs one training epoch
    :param total_train_batches: Number of batches to train on
    :return: mean_training_categorical_crossentropy_loss and mean_training_accuracy
    """
    total_c_loss = 0.
    total_accuracy = 0.
    # Create the optimizer
    model.to(device)

    with tqdm.tqdm(total=total_train_batches) as pbar:
        for i in range(total_train_batches):  # train epoch
            x_support_set, y_support_set, x_target, y_target = \
                data.get_batch()

            x_support_set = Variable(x_support_set).float()
            y_support_set = Variable(y_support_set, requires_grad=False).long()
            x_target = Variable(x_target).float()
            y_target = Variable(y_target, requires_grad=False).long()
            # y_support_set: Add extra dimension for the one_hot
            y_support_set = torch.unsqueeze(y_support_set, 2)
            sequence_length = y_support_set.size()[1]
            batch_size = y_support_set.size()[0]
            y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                            opt['classes_per_set']).zero_()
            y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
            y_support_set_one_hot = Variable(y_support_set_one_hot)

            x_target = x_target[:,-1,:,:]
            y_target = y_target[:,-1]


            acc, c_loss_value = model(x_support_set.to(device), y_support_set_one_hot.to(device),
                                      x_target.to(device), y_target.to(device))

            optim.zero_grad()
            c_loss_value.backward()
            optim.step()
            lr_scheduler.step()


            iter_out = "tr_loss: {}, tr_accuracy: {}".format(c_loss_value.item(), acc.item())
            pbar.set_description(iter_out)

            pbar.update(1)
            total_c_loss += c_loss_value.item()
            total_accuracy += acc.item()

        '''            self.total_train_iter += 1
            if self.total_train_iter % 2000 == 0:
                self.lr /= 2
                print("change learning rate", self.lr)'''

    avg_c_loss = total_c_loss / total_train_batches
    avg_acc = total_accuracy / total_train_batches
    return avg_c_loss, avg_acc

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def train_matching(data, opt, device, total_train_batches, optim, model, lr_scheduler):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_loss, train_acc = [], []
    best_acc = 0
    best_state = model.state_dict()  # Initialize with the current state of the model

    # Paths to save models in the current working directory
    cwd = os.getcwd()
    last_model_path = os.path.join(cwd, f'last_matching_net_.pth')
    best_model_path = os.path.join(cwd, 'best_matching_net.pth')

    for epoch in range(opt['epochs']):
        print('=== Epoch: {} ==='.format(epoch))
        avg_c_loss, avg_acc = run_training_epoch(data, opt, device, total_train_batches, optim, model, lr_scheduler)
        print('Avg train Loss: {}, Train Acc: {}'.format(avg_c_loss, avg_acc))
        train_loss.append(avg_c_loss)
        train_acc.append(avg_acc)

        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")
            best_acc = avg_acc
            best_state = model.state_dict()
        
    torch.save(model.state_dict(), last_model_path)
    print(f"Final model saved to {last_model_path}")

    for name in ['train_loss', 'train_acc']:
        save_list_to_file(os.path.join(cwd, name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc

def main_matching(opt):
    # Seed initialization
    torch.cuda.cudnn_enabled = False
    torch.manual_seed(opt['manual_seed'])
    torch.cuda.manual_seed(opt['manual_seed'])

    total_train_batches = opt['total_train_batches']
    model = MatchingNetwork(batch_size=opt['batch_size'],
                            num_channels=opt['channels'],
                            fce=opt['fce'],
                            num_classes_per_set=opt['classes_per_set'],
                            num_samples_per_class=opt['samples_per_class'])

    # Dataset and DataLoader
    data = SignalNShotTrainDataset(opt['train_dataset_path'],
                                      batch_size=opt['batch_size'],
                                      classes_per_set=opt['classes_per_set'], 
                                      samples_per_class=opt['samples_per_class'], 
                                      matching=True)

    # Model, Optimizer, and Scheduler
    device = opt['device']
    optim = torch.optim.Adam(params=model.parameters(), lr=opt['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optim,
        step_size=opt['lr_scheduler_step'],
        gamma=opt['lr_scheduler_gamma']
    )

    best_state, best_acc, train_loss, train_acc = train_matching(data, opt, device, total_train_batches, optim, model, lr_scheduler)
    
    #except:
    #    return model #Si on arrete prématurément, return le modèle

    model.load_state_dict(best_state)

    return model, best_acc, train_loss, train_acc


def normalize(x):
    """
    Normalise les signaux pour avoir une moyenne de 0 et un écart-type de 1.
    """
    mean = torch.mean(x)
    std = torch.std(x)
    return (x - mean) / std


def matching_n_shot(model, n=5, data_dir="../data", device="cuda:0"):

    n_test = 500
    x_enroll, y_enroll, _ = load_data(task="enroll", data_dir=data_dir)
    x_enroll, y_enroll = x_enroll.to(device), y_enroll.to(device)
    (x_n, y_n), idxs = get_n_per_classes(x_enroll, y_enroll, n=n)

    x_n = normalize(x_n)
    x_test, y_test, _ = load_data(task="test", data_dir=data_dir)
    x_test, y_test = x_test.to(device)[:n_test, :, :], y_test.to(device)[:n_test]
    x_test = normalize(x_test)

    x_support_set = Variable(x_n.unsqueeze(0)).float().to(device)
    y_support_set = Variable(y_n.unsqueeze(0)-y_n.min().item(),requires_grad=False).long().to(device)

    y_support_set = torch.unsqueeze(y_support_set, 2)

    sequence_length = y_support_set.size()[1]
    batch_size = y_support_set.size()[0]
    y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                    6).zero_().to(device)
    y_support_set_one_hot.scatter_(2, y_support_set.to(device), 1)
    y_support_set_one_hot = Variable(y_support_set_one_hot)

    accs = 0

    x_target = Variable(x_test).float().to(device)
    y_target = Variable(y_test-y_test.min(),requires_grad=False).long().to(device)

    model.to(device)

    with torch.no_grad():
        acc, loss = model(x_support_set.repeat(n_test, 1, 1, 1).to(device), y_support_set_one_hot.repeat(n_test,1, 1).to(device), x_target.to(device), y_target.to(device))
    accs += acc.item()

    return accs
