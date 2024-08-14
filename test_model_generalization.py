import argparse
import torch
from model import DTANet
import torch.nn as nn
from torch_geometric.data import DataLoader
from config import hyperparameter
from module_test import test_model
from torch_geometric.data import InMemoryDataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNDataset(InMemoryDataset):

    def __init__(self, root, train=False, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if train==False:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass


def test(dataset,model_nums,cv_folds):

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    assert dataset in ["EC50", "IC50", "Ki", "Kd", "kiba", "Davis"]
    print("Test in " + dataset)

    if dataset=="kiba" or dataset == "Davis":
        generalization_set_list = ['cold']
    else:
        generalization_set_list = ['ER', 'GPCR', 'IonChannel', 'Kinase']

    for x in generalization_set_list:
        CV_MSE_List, CV_CI_List, CV_R2_List = [], [], []
        for cv_fold in range(cv_folds):
            fpath = f'dataset/{dataset}/GeneralizationSet_{x}'
            test_data = GNNDataset(fpath, train=False)
            print('Number of GeneralizationSet {}: {}'.format(x,len(test_data)))

            test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0,drop_last = True)

            model = []
            criterion = nn.MSELoss().to(device)
            for i_model in range(model_nums):
                model.append(DTANet(hp, block_num=3, vocab_protein_size=25+1, vocab_drug_size=64+1, out_dim=1).to(device))
                '''DTA k_model train process is necessary'''
                try:
                    model[i_model].load_state_dict(torch.load(
                        f'./result/{dataset}/No_{cv_fold + 1}_fold cross validation/No_{i_model+1}_model' + '/valid_best_checkpoint.pth', map_location=torch.device(device)))
                except FileExistsError as e:
                    print('-'* 30 + 'ERROR' + '-'*30)
                    error_msg = 'Load pretrained model error: \n' + \
                                str(e) + \
                                '\n' + 'DTA k_model train process is necessary'
                    print(error_msg)
                    print('-' * 55)
                    exit(1)

            print(f"The ensemble model {cv_fold + 1}_fold tset in GeneralizationSet {x}:\n")
            testdataset_results, mse, ci, r2 = test_model(model, test_loader, criterion, device, dataset_class="Test",fold_num=model_nums)
            print()
            
            with open(f'result/{dataset}/result_generalization.txt','a')as file:
                file.write(f"The ensemble model {cv_fold + 1}_fold tset in GeneralizationSet {x}:\n")
                file.write('MSE(std):{:.4f}\n'.format(mse))  
                file.write('CI(std):{:.4f}\n'.format(ci))  
                file.write('R2(std):{:.4f}\n'.format(r2))
            
            CV_MSE_List.append(mse)
            CV_CI_List.append(ci)
            CV_R2_List.append(r2)

        Mse_mean, Mse_var = np.mean(CV_MSE_List), np.var(CV_MSE_List)
        Ci_mean, Ci_var = np.mean(CV_CI_List), np.var(CV_CI_List)
        R2_mean, R2_var = np.mean(CV_R2_List), np.var(CV_R2_List)
        print(f"The ensemble model 5_fold cross validation's results(GeneralizationSet_{dataset}):\n")
        print('MSE(std):{:.4f}({:.4f})'.format(Mse_mean, Mse_var))
        print('CI(std):{:.4f}({:.4f})'.format(Ci_mean, Ci_var))
        print('R2(std):{:.4f}({:.4f})'.format(R2_mean, R2_var))
        print()
        print()
        
        with open(f'result/{dataset}/result_generalization.txt','a')as file:
            file.write(f"The model 5_fold cross validation's mean results({dataset}_{x}):\n")
            file.write('MSE(std):{:.4f}({:.4f})\n'.format(Mse_mean, Mse_var))  
            file.write('CI(std):{:.4f}({:.4f})\n'.format(Ci_mean, Ci_var))  
            file.write('R2(std):{:.4f}({:.4f})\n'.format(R2_mean, R2_var))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DTANet',description='Drug-protein Affinity',epilog='Model config set by config.py')
    parser.add_argument('dataSetName', choices=["EC50", "IC50", "Ki", "Kd","kiba","Davis"],help='Enter which dataset to use for the experiment')
    parser.add_argument('-n','--model_nums', type=int, default=5, help='Sets the number of ensemble models, the default is 5')
    parser.add_argument('-cv_f','--cross_validation_folds', type=int, default=5, help='Set the number of folds for cross validation, the default is 5')

    args = parser.parse_args()

    test(dataset=args.dataSetName, model_nums=args.model_nums, cv_folds=args.cross_validation_folds)

