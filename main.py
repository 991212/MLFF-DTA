import argparse
from train import run_model


parser = argparse.ArgumentParser(prog='DTANet',description='Drug-protein Affinity',epilog='Model config set by config.py')

parser.add_argument('dataSetName', choices=["EC50","Kd","Davis"],help='Enter which dataset to use for the experiment')
parser.add_argument('-n','--model_nums', type=int, default=5, help='Sets the number of ensemble models, the default is 5')
parser.add_argument('-cv_f','--cross_validation_folds', type=int, default=5, help='Set the number of folds for cross validation, the default is 5')


args = parser.parse_args()

run_model(dataset=args.dataSetName, model_nums=args.model_nums, cross_validation_folds=args.cross_validation_folds)
