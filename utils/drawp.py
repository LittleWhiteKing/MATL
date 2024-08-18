
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# file_path = '/home/fanjinli/myproject/aDNA_TFBSs/acc.xlsx'
# df = pd.read_excel(file_path)
#
# sns.set(style="white")

# plt.figure(figsize=(12, 6))
# sns.boxplot(data=df)
# plt.title('Performance Metrics Distribution')
# plt.ylabel('Value')
# plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# file_path = '/home/fanjinli/myproject/aDNA_TFBSs/accxiaorong.xlsx'
# df = pd.read_excel(file_path)

# # sns.set(style="")
# plt.figure(figsize=(4.5, 4))

# sns.boxplot(data=df, flierprops={'marker':'o', 'markerfacecolor':'black', 'markersize':3, 'linestyle':'none'})
# plt.gca().tick_params(axis='x', which='both', direction='in', length=3, width=1)
# plt.gca().tick_params(axis='y', which='both', direction='in', length=3, width=1)
# plt.xticks(fontsize=10, fontweight='bold', rotation=15)
# plt.yticks(fontsize=8, fontweight='bold')
# # plt.title('ACC',fontweight='bold')
# plt.ylabel('ACC', fontweight='bold', fontsize=13)
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = '/home/fanjinli/myproject/aDNA_TFBSs/acc.xlsx'
df = pd.read_excel(file_path)




plt.figure(figsize=(4.5, 4))
sns.violinplot(data=df, inner="box", linewidth=1.3)
ax = plt.gca()


plt.setp(ax.get_xticklabels(), fontname='Times New Roman', fontweight='bold',fontsize=8.7,rotation=15)
plt.setp(ax.get_yticklabels(), fontname='Times New Roman', fontweight='bold',fontsize=10)
plt.gca().tick_params(axis='x', which='both', direction='in', length=3, width=1)
plt.gca().tick_params(axis='y', which='both', direction='in', length=3, width=1)
ax.set_ylabel('ACC', fontweight='bold', fontsize=13, fontname='Times New Roman')


plt.show()


# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#

# file_path = '/home/fanjinli/myproject/other_methed/result_seq_shape.xlsx'
# df = pd.read_excel(file_path)
#

# categories = df['tfs'].tolist()
# seq_shape = df['shape_seq'].tolist()
# sequeence = df['seq'].tolist()
#

# data_with_labels = list(zip(categories, seq_shape, sequeence))

# sorted_data = sorted(data_with_labels, key=lambda x: x[1], reverse=True)

# sorted_categories, sorted_sequence_data, sorted_shape_data = zip(*sorted_data)

# fig, ax = plt.subplots(figsize=(14, 8))
# x = np.arange(len(sorted_categories))
# width = 0.6
#
# bars1 = ax.bar(x, sorted_sequence_data, width, label='Sequence_Shape', color='blue')
# bars2 = ax.bar(x, sorted_shape_data, width, label='Sequence', color='orange')

# ax.set_ylabel('ACC', fontweight='bold', fontsize=30)
# ax.set_xticks(x)
# ax.set_xticklabels(sorted_categories, rotation=40, fontweight='bold', fontsize=13)
# plt.setp(ax.get_yticklabels(), fontname='Times New Roman', fontweight='bold', fontsize=15)
# ax.legend()
# legend = ax.legend(fontsize=15)
# for text in legend.get_texts():
#     text.set_fontweight('bold')
#

# # ax.set_title('ACC', fontweight='bold', fontsize=15)  # 去掉这行
#

# plt.tight_layout()
# plt.show()
#
