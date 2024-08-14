import copy
from metrics import *
from tqdm import tqdm
from utils.utils import *
from prefetch_generator import BackgroundGenerator
def test_precess(model, pbar, criterion,device,fold_num):
    if isinstance(model, list):
        for item in model:
            item.eval()
    else:
        model.eval()

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    pred_list = []
    label_list = []


    with torch.no_grad():

        for i, data in pbar:
            '''data preparation '''
            torch.cuda.empty_cache()
            data = data.to(device) #(18888, 18888)

            if isinstance(model, list):
                # predicted_scores = torch.zeros((512)).to(device)
                predicted_scores = torch.zeros(1).to(device)
                # print("predicted_scores:",predicted_scores.shape)
                # print("model:", model[0](data).shape, model[0](data))
                for j in range(len(model)):
                    tmp_data = copy.deepcopy(data)
                    tmp_data.to(device)
                    predicted_scores = predicted_scores + model[j](tmp_data)
                    # print("第",j,"个模型：",type(predicted_scores),predicted_scores.shape)
                predicted_scores = predicted_scores / fold_num
            else:
                predicted_scores = model(data)


            label = data.y
            # print("predicted_scores:",type(predicted_scores),predicted_scores.shape,predicted_scores.view(-1))
            # print("label",type(label),label.shape,label.view(-1))
            loss = criterion(predicted_scores.view(-1), label.view(-1))
            cindex = get_cindex(label.detach().cpu().numpy().reshape(-1),
                                      predicted_scores.detach().cpu().numpy().reshape(-1))

            pred_list.append(predicted_scores.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

            running_loss.update(loss.item(), label.size(0))
            running_cindex.update(cindex, label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    epoch_cindex = get_cindex(label, pred)
    epoch_r2 = get_rm2(label, pred)
    epoch_loss = running_loss.get_average()
    running_loss.reset()

    return epoch_loss,epoch_cindex,epoch_r2

def test_model(model, dataset_loader, criterion, device, dataset_class="Train", fold_num=1):
    test_pbar = tqdm(enumerate(BackgroundGenerator(dataset_loader)),total=len(dataset_loader))
    loss,cindex,r2 = test_precess(model, test_pbar, criterion, device, fold_num)

    results = '{}: Loss:{:.5f};ci:{:.5f};r2:{:.5f}.' \
        .format(dataset_class, loss, cindex, r2)
    print(results)

    return results,loss,cindex,r2
