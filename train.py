import random
import torch.optim as optim
from dataset import *
import torch.nn as nn
from model import DTANet
from config import hyperparameter
from utils.ShowResult import show_result
from module_test import *
from prefetch_generator import BackgroundGenerator
from utils.EarlyStoping import EarlyStopping
from log.train_logger import TrainLogger
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Initialize tensorboard log folder'''
writer = SummaryWriter('./tensorboard_log')

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()
    running_cindex = AverageMeter()

    for data in dataloader:
        torch.cuda.empty_cache()
        data = data.to(device)

        with torch.no_grad():
            predicted_value = model(data)
            loss = criterion(predicted_value.view(-1), data.y.view(-1))
            cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1),
                                      predicted_value.detach().cpu().numpy().reshape(-1))

            running_loss.update(loss.item(), data.y.size(0))
            running_cindex.update(cindex, data.y.size(0))

    epoch_loss = running_loss.get_average()
    epoch_cindex = running_cindex.get_average()
    running_loss.reset()
    running_cindex.reset()

    model.train()

    return epoch_loss,epoch_cindex

def run_model(dataset,model_nums,cross_validation_folds):

    '''init hyperparameters'''
    hp = hyperparameter()

    random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed_all(hp.seed)

    CV_MSE_List, CV_CI_List, CV_R2_List = [], [], []
    for cv_fold in range(cross_validation_folds):
    # for cv_fold in range(cross_validation_folds):
        print('*' * 50, 'No_', cv_fold + 1, '_fold cross validation', '*' * 50)
        data_root = 'dataset'
        data_folder = 'No_{}_fold cross validation data'.format(cv_fold+1)

        #dataset\Kd\No_1_fold cross validation data
        fpath = os.path.join(data_root, dataset,data_folder)
        
        # '''split dataset to train set and test set'''
        train_set = GNNDataset(fpath, train=True)
        test_set = GNNDataset(fpath, train=False)

        # '''split dataset to train set and validation set'''
        train_val_folds = train_set.split_train_val_folds(model_nums,hp.seed)

        '''metrics'''
        MSE_List,CI_List, R2_List = [], [], []


        for i_model, (train_data, valid_data) in enumerate(train_val_folds):
        # for i_model in range(0, 1):
            print('*' * 40, 'No_', i_model + 1, '_model', '*' * 40)

            #读取每个model的数据进行训练
            # train_data = torch.load(os.path.join(fpath, f'model{i_model + 1}', 'train_data.pt'))
            # valid_data = torch.load(os.path.join(fpath, f'model{i_model + 1}', 'valid_data.pt'))

            print('Number of Train set: {}'.format(len(train_data)))
            print('Number of Val set: {}'.format(len(valid_data)))
            train_loader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True,num_workers=0)
            valid_loader = DataLoader(valid_data, batch_size=hp.batch_size, shuffle=False,num_workers=0)
            test_loader = DataLoader(test_set,batch_size=hp.batch_size,shuffle=False,num_workers=0)


            """ create model"""
            model = DTANet(hp,block_num=3, vocab_protein_size=25 + 1, vocab_drug_size=64+1, out_dim=1)

            # if torch.cuda.device_count()>1:
            #     print("Training with multiple gpu...")
            #     model = nn.DataParallel(model)

            model = model.to(device)

            """Initialize weights"""
            weight_p, bias_p = [], []
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for name, p in model.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]

            """create optimizer and scheduler(Loop learning rate scheduling policy)"""
            """Set up an AdamW optimizer with weight attenuation and a cyclic learning rate scheduler for training the model.
            The optimizer will update the weights of the model using the adaptive learning rate, 
            The scheduler will periodically adjust the learning rate to change within the specified range during training."""
            optimizer = optim.AdamW(
                [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}],
                lr=hp.learning_rate)

            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.learning_rate, max_lr=hp.learning_rate * 10,
                                                    cycle_momentum=False,
                                                    step_size_up=len(train_data) // hp.batch_size)
            criterion = nn.MSELoss().to(device)

            """Output files"""
            save_path = "./result/" + dataset + "/No_{}_fold cross validation/".format(cv_fold + 1) + "/No_{}_model".format(i_model + 1)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            params = dict(
                save_dir=save_path,
                dataset=dataset,
                lr=hp.learning_rate,
                batch_size=hp.batch_size
            )
            logger = TrainLogger(params)
            logger.info(__file__)

            early_stopping = EarlyStopping(savepath=save_path, patience=hp.patience, verbose=True, delta=0)

            """Start training."""
            print('Training...')
            for epoch in range(1,hp.epoch+1):
                if early_stopping.early_stop == True:
                    logger.info(f"early stop in epoch {epoch-1}")
                    break
                train_pbar = tqdm(enumerate(BackgroundGenerator(train_loader)),
                    total=len(train_loader))

                '''train'''
                running_loss = AverageMeter()
                running_cindex = AverageMeter()
                running_best_mse = BestMeter("min")

                model.train()
                for train_i,train_data in train_pbar:
                    torch.cuda.empty_cache()
                    train_data = train_data.to(device)

                    train_predicted_value = model(train_data)

                    train_loss = criterion(train_predicted_value.view(-1), train_data.y.view(-1))
                    train_cindex = get_cindex(train_data.y.detach().cpu().numpy().reshape(-1), train_predicted_value.detach().cpu().numpy().reshape(-1))

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    scheduler.step()

                    running_loss.update(train_loss.item(), train_data.y.size(0))
                    running_cindex.update(train_cindex, train_data.y.size(0))

                train_loss_a_epoch = running_loss.get_average()
                train_cindex_a_epoch = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()

                '''valid'''
                valid_pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)),total=len(valid_loader))
                model.eval()

                with torch.no_grad():
                    for valid_i,valid_data in valid_pbar:
                        torch.cuda.empty_cache()
                        valid_data = valid_data.to(device)
                        valid_predicted_value = model(valid_data)

                        valid_loss = criterion(valid_predicted_value.view(-1), valid_data.y.view(-1))
                        valid_cindex = get_cindex(valid_data.y.detach().cpu().numpy().reshape(-1),
                                                  valid_predicted_value.detach().cpu().numpy().reshape(-1))

                        running_loss.update(valid_loss.item(), valid_data.y.size(0))
                        running_cindex.update(valid_cindex, valid_data.y.size(0))

                    valid_loss_a_epoch = running_loss.get_average()
                    valid_cindex_a_epoch = running_cindex.get_average()
                    running_loss.reset()
                    running_cindex.reset()


                msg = "epoch:%d, train_loss:%.5f, train_ci:%.5f, valid_loss:%.5f,valid_ci:%.5f" % (
                epoch, train_loss_a_epoch, train_cindex_a_epoch, valid_loss_a_epoch,valid_cindex_a_epoch)
                logger.info(msg)

                '''writer tarin_loss,train_ci,valid_loss,valid_ci'''
                writer.add_scalar(f'{dataset}/No_{cv_fold+1}fold CV/No_{i_model + 1}_model/train_loss', train_loss_a_epoch, global_step=epoch)
                writer.add_scalar(f'{dataset}/No_{cv_fold+1}fold CV/No_{i_model + 1}_model/train_ci', train_cindex_a_epoch, global_step=epoch)
                writer.add_scalar(f'{dataset}/No_{cv_fold+1}fold CV/No_{i_model + 1}_model/valid_loss', valid_loss_a_epoch, global_step=epoch)
                writer.add_scalar(f'{dataset}/No_{cv_fold+1}fold CV/No_{i_model + 1}_model/valid_ci', valid_cindex_a_epoch, global_step=epoch)


                '''save checkpoint and make decision when early stop'''
                early_stopping(valid_loss_a_epoch,valid_cindex_a_epoch,model, epoch)

            '''load best checkpoint'''
            model.load_state_dict(torch.load(
                early_stopping.savepath + '/valid_best_checkpoint.pth'))

            '''test model'''
            trainset_test_stable_results, _, _, _ = test_model(model, train_loader, criterion, device, dataset_class="Train", fold_num=1)
            validset_test_stable_results, _, _, _= test_model(model, valid_loader, criterion, device, dataset_class="Valid", fold_num=1)
            testset_test_stable_results, loss,cindex,r2 = test_model(model,test_loader, criterion, device, dataset_class="Test", fold_num=1)
            MSE_List.append(loss)
            CI_List.append(cindex)
            R2_List.append(r2)

            '''write MSE and R2 of the k_model'''
            writer.add_scalar(f'{dataset}/No_{cv_fold+1}/MSE', loss, global_step=i_model + 1)
            writer.add_scalar(f'{dataset}/No_{cv_fold+1}/CI', cindex, global_step=i_model + 1)
            writer.add_scalar(f'{dataset}/No_{cv_fold+1}/R2', r2, global_step=i_model + 1)

            with open(save_path + '/' + f"The results of No_{i_model+1}_model.txt", 'a') as f:
                f.write("Test the stable model" + '\n')
                f.write(trainset_test_stable_results + '\n')
                f.write(validset_test_stable_results + '\n')
                f.write(testset_test_stable_results + '\n')


        "Show the mean results of the k_fold model"
        show_result(dataset,cv_fold, MSE_List, CI_List, R2_List, tag='0')


        '''ensemble model'''
        print('Number of Test set: {}'.format(len(test_set)))

        test_loader = DataLoader(test_set, batch_size=hp.batch_size, shuffle=False, num_workers=0, drop_last = True)

        model = []
        criterion = nn.MSELoss().to(device)
        for i_model in range(model_nums):
            model.append(DTANet(hp,block_num=3, vocab_protein_size=25 + 1, vocab_drug_size=64+1, out_dim=1).to(device))
            '''DTA K-Fold train process is necessary'''
            try:
                model[i_model].load_state_dict(torch.load(
                    f'./result/{dataset}/No_{cv_fold+1}_fold cross validation/No_{i_model+1}_model' + '/valid_best_checkpoint.pth', map_location=torch.device(device)))
            except FileExistsError as e:
                print('-'* 30 + 'ERROR' + '-'*30)
                error_msg = 'Load pretrained model error: \n' + \
                            str(e) + \
                            '\n' + 'DTA k_fold train process is necessary'
                print(error_msg)
                print('-' * 55)
                exit(1)

        testdataset_results, mse, ci, r2 = test_model(model, test_loader, criterion, device, dataset_class="Test", fold_num=model_nums)
        show_result(dataset,cv_fold, mse, ci, r2, tag='1')
        CV_MSE_List.append(mse)
        CV_CI_List.append(ci)
        CV_R2_List.append(r2)
    show_result(dataset,5, CV_MSE_List, CV_CI_List, CV_R2_List, tag='2')
