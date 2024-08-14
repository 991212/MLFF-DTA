import argparse
import torch
from model import DTANet
import torch.nn as nn
from torch_geometric.data import DataLoader
from dataset import GNNDataset
from config import hyperparameter
from module_test import test_model
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_ensemble_model(dataset,model_nums,cv_folds):

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    assert dataset in ["EC50", "IC50", "Ki", "Kd"]
    print("Test in " + dataset)

    CV_MSE_List, CV_CI_List, CV_R2_List = [], [], []
    for cv_fold in range(cv_folds):
        fpath = 'dataset/{}/No_{}_fold cross validation data'.format(dataset,cv_fold+1)
        test_data = GNNDataset(fpath, train=False)
        print(f'Number of No_{cv_fold + 1 }_fold Test set: {len(test_data)}')

        test_loader = DataLoader(test_data, batch_size=hp.batch_size, shuffle=False, num_workers=0,drop_last = True)

        model = []
        criterion = nn.MSELoss().to(device)
        for i_model in range(model_nums):
            model.append(DTANet(hp, block_num=3, vocab_protein_size=25+1, vocab_drug_size=64+1, out_dim=1).to(device))
            '''DTA k_model train process is necessary'''
            try:
                model[i_model].load_state_dict(torch.load(
                    f'./result/{dataset}/No_{cv_fold+1}_fold cross validation/No_{i_model+1}_model' + '/valid_best_checkpoint.pth', map_location=torch.device(device)))
            except FileExistsError as e:
                print('-'* 30 + 'ERROR' + '-'*30)
                error_msg = 'Load pretrained model error: \n' + \
                            str(e) + \
                            '\n' + 'DTA k_model train process is necessary'
                print(error_msg)
                print('-' * 55)
                exit(1)

        print(f"The ensemble model {cv_fold + 1}_fold cross validation:")
        testdataset_results, mse, ci, r2 = test_model(model, test_loader, criterion, device, dataset_class="Test", fold_num=model_nums)
        print()
        CV_MSE_List.append(mse)
        CV_CI_List.append(ci)
        CV_R2_List.append(r2)

    Mse_mean, Mse_var = np.mean(CV_MSE_List), np.var(CV_MSE_List)
    Ci_mean, Ci_var = np.mean(CV_CI_List), np.var(CV_CI_List)
    R2_mean, R2_var = np.mean(CV_R2_List), np.var(CV_R2_List)
    print()
    print(f"The ensemble model 5_fold cross validation's results({dataset}):")
    print('MSE(std):{:.4f}({:.4f})'.format(Mse_mean, Mse_var))
    print('CI(std):{:.4f}({:.4f})'.format(Ci_mean, Ci_var))
    print('R2(std):{:.4f}({:.4f})'.format(R2_mean, R2_var))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DTANet',description='Drug-protein Affinity',epilog='Model config set by config.py')
    parser.add_argument('dataSetName', choices=["EC50", "IC50", "Ki", "Kd","kiba"],help='Enter which dataset to use for the experiment')
    parser.add_argument('-n','--model_nums', type=int, default=5, help='Sets the number of ensemble models, the default is 5')
    parser.add_argument('-cv_f','--cross_validation_folds', type=int, default=5, help='Set the number of folds for cross validation, the default is 5')

    args = parser.parse_args()

    run_ensemble_model(dataset=args.dataSetName, model_nums=args.model_nums, cv_folds=args.cross_validation_folds)




