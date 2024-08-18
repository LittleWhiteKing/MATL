import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.optim as optim
import torch.utils.data as loader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import random_split
from utils.main_encode import *
from utils.adjust_learning import *
from all_model.MATL import *
import pandas as pd


class main:
    def __init__(self, model, model_name='my_model', batch_size=32, epochs=15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1.0e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5)
        self.loss_function = nn.BCELoss()
        self.batch_size = batch_size
        self.epochs = epochs

    def learn(self, TrainLoader, ValidateLoader):
        path = '/home/fanjinli/myproject/aDNA_TFBSs/model_pth'
        best_accuracy = 0.0
        best_model_path = None

        for epoch in range(self.epochs):
            print(f"----Starting epoch {epoch + 1}/{self.epochs}-----")
            self.model.train()
            for data in TrainLoader:
                self.optimizer.zero_grad()
                seq, shape, label = data

                output = self.model(seq.unsqueeze(1).to(self.device), shape.unsqueeze(1).to(self.device))
                loss = self.loss_function(output, label.float().to(self.device))
                loss.backward()
                self.optimizer.step()
            adjust_learning_rate(optimizer=self.optimizer,
                                 current_epoch=epoch,
                                 max_epoch=self.epochs,
                                 lr_min=2.0e-5,
                                 lr_max=1.0e-3,
                                 warmup=True)

            print(f"！！！Starting epoch {epoch + 1}/{self.epochs}！！!---train loss:---! {loss.item():.4f}")

            valid_losses = []
            valid_outputs = []
            valid_labels = []
            self.model.eval()
            with torch.no_grad():
                for valid_seq, valid_shape, valid_label in ValidateLoader:
                    valid_output = self.model(valid_seq.unsqueeze(1).to(self.device),
                                              valid_shape.unsqueeze(1).to(self.device))
                    valid_losses.append(self.loss_function(valid_output, valid_label.float().to(self.device)).item())
                    valid_outputs.extend(valid_output.round().cpu().numpy())
                    valid_labels.extend(valid_label.cpu().numpy())

                valid_loss_avg = torch.mean(torch.Tensor(valid_losses))
                self.scheduler.step(valid_loss_avg)
                accuracy = accuracy_score(valid_labels, valid_outputs)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    if best_model_path:
                        os.remove(best_model_path)
                    best_model_path = os.path.join(path, f"{self.model_name}_epoch-{epoch}_acc-{accuracy:.4f}.pth")
                    self.best_model_path = best_model_path
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"Epoch {epoch}: New best model saved with accuracy {accuracy:.4f}")

        print('\n---Finish Learn---\n')

    def inference(self, TestLoader):
        self.model.load_state_dict(torch.load(self.best_model_path, map_location='cpu'))
        predicted_value = []
        ground_label = []
        self.model.eval()
        for seq, shape, label in TestLoader:
            output = self.model(seq.unsqueeze(1).to(self.device), shape.unsqueeze(1).to(self.device))
            """ To scalar"""
            predicted_value.append(output.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())
            ground_label.append(label.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())
        print('\n---Finish Inference---\n')
        return predicted_value, ground_label

    def measure(self, predicted_value, ground_label):
        accuracy = accuracy_score(y_pred=np.array(predicted_value).round(), y_true=ground_label)
        roc_auc = roc_auc_score(y_score=predicted_value, y_true=ground_label)
        precision, recall, _ = precision_recall_curve(probas_pred=predicted_value, y_true=ground_label)
        pr_auc = auc(recall, precision)
        print('\n---Finish Measure---\n')
        return accuracy, roc_auc, pr_auc

    def cross_validate(self, samples_file_name, k_folds=5):
        dataset = SSDataset_690(samples_file_name, False)  # 假设这是你的数据集类
        kfold = KFold(n_splits=k_folds, shuffle=False)
        accuracies = []
        roc_aucs = []
        pr_aucs = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f'Fold {fold + 1}/{k_folds}')
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            TrainLoader = loader.DataLoader(dataset=train_subset,
                                            drop_last=True,
                                            batch_size=self.batch_size, shuffle=True,
                                            num_workers=0)
            ValidateLoader = loader.DataLoader(dataset=val_subset,
                                               drop_last=True,
                                               batch_size=self.batch_size, shuffle=False,
                                               num_workers=0)
            self.learn(TrainLoader, ValidateLoader)
            predicted_value, ground_label = self.inference(ValidateLoader)
            predicted_value = np.array(predicted_value).flatten()
            ground_label = np.array(ground_label).flatten()
            accuracy, roc_auc, pr_auc = self.measure(predicted_value, ground_label)
            accuracies.append(accuracy)
            roc_aucs.append(roc_auc)
            pr_aucs.append(pr_auc)
            print(f'Fold {fold + 1}/{k_folds} - Accuracy: {accuracy}, ROC AUC: {roc_auc}, PR AUC: {pr_auc}')
        print('\n--- Cross Validation Finished ---\n')
        print(f'Average Accuracy: {np.mean(accuracies)}, Average ROC AUC: {np.mean(roc_aucs)}, Average PR AUC: {np.mean(pr_aucs)}')
        return accuracies, roc_aucs, pr_aucs

    def run(self, samples_file_name, ratio=0.8):
        Train_Validate_Set = SSDataset_690(samples_file_name, False)
        """divide Train samples and Validate samples"""
        Train_Set, Validate_Set = random_split(
            dataset=Train_Validate_Set,
            lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                     len(Train_Validate_Set) - math.ceil(len(Train_Validate_Set) * ratio)],
            generator=torch.Generator().manual_seed(0))
        TrainLoader = loader.DataLoader(dataset=Train_Set,
                                        drop_last=True,
                                        batch_size=self.batch_size, shuffle=True,
                                        num_workers=0)
        ValidateLoader = loader.DataLoader(dataset=Validate_Set,
                                           drop_last=True,
                                           batch_size=self.batch_size, shuffle=False,
                                           num_workers=0)
        TestLoader = loader.DataLoader(dataset=SSDataset_690(samples_file_name, True),
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=0)
        self.learn(TrainLoader, ValidateLoader)
        predicted_value, ground_label = self.inference(TestLoader)
        accuracy, roc_auc, pr_auc = self.measure(predicted_value, ground_label)
        print('\n---Finish Run---\n')
        print('\nAccuracy: {:f}, ROC AUC: {:f}, PR AUC: {:f}'.format(accuracy, roc_auc, pr_auc))
        return accuracy, roc_auc, pr_auc


# def run_modeling_for_datasets(names_file):

#     dataset_names = pd.read_excel(names_file)  
#     results = [] 

#     for index, row in dataset_names.iterrows():
#         dataset_name = row[0] 
#         print(f"{dataset_name}")


#         accuracy, roc_auc, pr_auc = model_instance.run(samples_file_name=dataset_name)

#         results.append({
#             'Dataset': dataset_name,
#             'Accuracy': accuracy,
#             'ROC AUC': roc_auc,
#             'PR AUC': pr_auc
#         })
#         print(f"{dataset_name}, ACC：{accuracy}, ROC AUC：{roc_auc}, PR AUC：{pr_auc}")

#     results_df = pd.DataFrame(results)
#     results_df.to_csv('modeling_result_DSAC.csv', index=False)
#     print("modeling_result.csv 文件中。")
#     return results_df
# run_modeling_for_datasets('/home/fanjinli/myproject/aDNA_TFBSs/Dataset.xlsx')
# #

Train = main(model=MATL())
Train.run(samples_file_name='wgEncodeAwgTfbsBroadHuvecEzh239875UniPk')


