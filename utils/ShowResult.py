import numpy as np
import os


def show_result(dataset, cv_flod, MSE_List, CI_List, R2_List, tag='0'):
    Mse_mean, Mse_var = np.mean(MSE_List), np.var(MSE_List)
    Ci_mean, Ci_var = np.mean(CI_List), np.var(CI_List)
    R2_mean, R2_var = np.mean(R2_List), np.var(R2_List)

    if tag == '0':
        print("No_{}_fold cross-validation, k_models' mean results (no ensemble):".format(cv_flod+1))
        filepath = "./result/{}".format(dataset)
        filename = "k_models's mean results (no ensemble,each cross-validation).txt"
        file_result = os.path.join(filepath,filename)
        with open(file_result, 'a') as f:
            f.write('No_{}_fold cross-validation:   MSE(std):{:.4f}({:.4f})   CI(std):{:.4f}({:.4f})   R2(std):{:.4f}({:.4f})'.format(cv_flod+1, Mse_mean, Mse_var, Ci_mean, Ci_var, R2_mean, R2_var) + '\n' + '\n')

    elif tag == '1':
        print("No_{}_fold cross-validation, ensemble model's results:".format(cv_flod + 1))
        filepath = "./result/{}".format(dataset)
        filename = "ensemble model's results(each fold cross-validation).txt"
        file_result = os.path.join(filepath,filename)
        with open(file_result, 'a') as f:
            f.write('No_{}_fold cross validation:   MSE(std):{:.4f}({:.4f})   CI(std):{:.4f}({:.4f})   R2(std):{:.4f}({:.4f})'.format(cv_flod+1, Mse_mean, Mse_var, Ci_mean, Ci_var, R2_mean, R2_var) + '\n' + '\n')

    else:
        print("The ensemble model cross validation's mean results:")
        filepath = "./result/{}".format(dataset)
        filename = "ensemble model cross-validation's mean results.txt"
        file_result = os.path.join(filepath, filename)
        with open(file_result, 'a') as f:
            f.write("The ensemble model cross validation's mean results:" + '\n')
            f.write('MSE(std):{:.4f}({:.4f})'.format(Mse_mean, Mse_var) + '\n')
            f.write('CI(std):{:.4f}({:.4f})'.format(Ci_mean, Ci_var) + '\n')
            f.write('R2(std):{:.4f}({:.4f})'.format(R2_mean, R2_var) + '\n')


    print('MSE(std):{:.4f}({:.4f})'.format(Mse_mean, Mse_var))
    print('CI(std):{:.4f}({:.4f})'.format(Ci_mean, Ci_var))
    print('R2(std):{:.4f}({:.4f})'.format(R2_mean, R2_var))

