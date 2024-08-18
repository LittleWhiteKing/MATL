import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.optim as optim
import torch.utils.data as loader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import random_split
from utils.main_encode import *
from aDNA_TFBSs.all_model.MATL import *
import pandas as pd
from utils.adjust_learning import *


class main:
    def __init__(self, model, model_name='my_model', batch_size=32, epochs=15, pretrained_model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        if pretrained_model_path:
            self.model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
            print("Loaded pretrained model from:", pretrained_model_path)
        self.optimizer = optim.AdamW(self.model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5)
        self.loss_function = nn.BCELoss()
        self.batch_size = batch_size
        self.epochs = epochs

    def load_selected_weights(self, pretrained_model_path):
        pretrained_dict = torch.load(pretrained_model_path, map_location=self.device)
        model_dict = self.model.state_dict()
        selected_weights = {k: v for k, v in pretrained_dict.items() if 'transformer_shape' in k or 'lstm' in k}
        model_dict.update(selected_weights)
        self.model.load_state_dict(model_dict)
        self.reset_parameters_except('transformer_shape', 'lstm')

    def reset_parameters_except(self, *exclude_layers):
        for name, module in self.model.named_children():
            if not any(layer in name for layer in exclude_layers):
                self.reset_parameters(module)

    def reset_parameters(self, module):
        for layer in module.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def learn(self, TrainLoader, ValidateLoader):
        path = '/home/fanjinli/myproject/aDNA_TFBSs/model_pth'

        best_accuracy = 0.0  # 初始化最佳准确率为0
        best_model_path = None  # 初始化最佳模型路径为None

        # 开始模型的循环训练

        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()
                ProgressBar.set_description("Epoch %d" % epoch)
                seq, shape, label = data
                output = self.model(seq.unsqueeze(1).to(self.device), shape.unsqueeze(1).to(self.device))
                loss = self.loss_function(output, label.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())
                loss.backward()
                self.optimizer.step()
            adjust_learning_rate(optimizer=self.optimizer,
                                 current_epoch=epoch,
                                 max_epoch=self.epochs,
                                 lr_min=2.0e-5,
                                 lr_max=1.0e-3,
                                 warmup=True)

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
                                        batch_size=self.batch_size,shuffle=True,
                                        num_workers=0)
        # 验证集的作用是调整学习率。
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

#
# def run_modeling_for_datasets(names_file, pretrained_model_path):
#     """
#     进行所有数据的运行，单独测试这个模型，得到的结果放到excel中
#     """
#     # 读取数据集名称
#     dataset_names = pd.read_excel(names_file)  # 修改此处以读取 Excel 文件
#     results = []  # 用于存储每个数据集的结果
#     # 遍历每个数据集名称
#     for index, row in dataset_names.iterrows():
#         dataset_name = row[0]  # 假设数据集名称在第一列
#         print(f"开始建模：{dataset_name}")
#         # 每个数据集初始化一个新的模型实例，加载预训练模型
#         model_instance = main(model=MTCCA(), pretrained_model_path=pretrained_model_path)
#         # 运行模型
#         accuracy, roc_auc, pr_auc = model_instance.run(samples_file_name=dataset_name)
#         # 存储结果
#         results.append({
#             'Dataset': dataset_name,
#             'Accuracy': accuracy,
#             'ROC AUC': roc_auc,
#             'PR AUC': pr_auc
#         })
#         print(f"模型建模完成：{dataset_name}, 准确率：{accuracy}, ROC AUC：{roc_auc}, PR AUC：{pr_auc}")
#     # 可以将结果保存到CSV文件或返回
#     results_df = pd.DataFrame(results)
#     results_df.to_csv('modeling_results_trans710.csv', index=False)
#     print("所有建模过程已完成，结果已保存到 modeling_results_trans710 文件中。")
#     return results_df
#
# pretrained_path = '/home/fanjinli/myproject/aDNA_TFBSs/model_pth/my_model_epoch-3_acc-0.7697.pth'
# run_modeling_for_datasets('/home/fanjinli/myproject/aDNA_TFBSs/Dataset.xlsx', pretrained_path)
#
Train = main(model=MATL(), pretrained_model_path='/home/fanjinli/myproject/aDNA_TFBSs/model_pth/my_model_epoch-4_acc-0.9290.pth')
Train.run(samples_file_name='wgEncodeAwgTfbsBroadNhekPol2bUniPk')
