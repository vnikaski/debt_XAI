import pandas as pd
import numpy as np

dic = {'Kids': [['X1', 'X2', 'X3'], ['X64', 'X65', 'X66'], ['X169', 'X170', 'X171']],
       'Health': [['X225', 'X226', 'X227'], ['X8', 'X9', 'X10']],
       'ATM': [['X15', 'X16', 'X17']],
       'Home': [['X36', 'X37', 'X38'], ['X43', 'X44', 'X45'], ['X71', 'X72', 'X73'], ['X120', 'X121', 'X122'], ['X134', 'X135', 'X136'],
                ['X141', 'X142', 'X143'], ['X183', 'X184', 'X185'], ['X232', 'X233', 'X234']],
       'Food': [['X50', 'X51', 'X52'], ['X57', 'X58', 'X59'], ['X85', 'X86', 'X87']], 
       'Other': [['X92', 'X93', 'X94'], ['X99', 'X100', 'X101'], ['X106', 'X107', 'X108'], ['X148', 'X149', 'X150'], ['X78', 'X79', 'X80'], 
                 ['X155', 'X156', 'X157'], ['X176', 'X177', 'X178'], ['X218', 'X219', 'X220'], ['X239', 'X240', 'X241'], 
                 ['X29', 'X30', 'X31']], 
       'Transport': [['X113', 'X114', 'X115'], ['X127', 'X128', 'X129'], ['X190', 'X191', 'X192'], ['X197', 'X198', 'X199'], 
                     ['X204', 'X205', 'X206']], 
       'Insurance': [['X211', 'X212', 'X213']], 
       'Allowance': [['X246', 'X247', 'X248'], ['X257', 'X258', 'X259'], ['X260', 'X261', 'X262'], ['X274', 'X275', 'X276']], 
       'Earnings': [['X271', 'X272', 'X273'], ['X288', 'X289', 'X290']], 
       'All_prop': [['X281', 'X282', 'X283'], ['X22', 'X23', 'X24']]
      }
       

def new_feats(df):
    new_df = {}
    for new_feat, old_feats in dic.items():
        if new_feat != 'All_prop':
            temp = np.zeros((3, len(df)))
            for old_feat in old_feats:
                for i in range(3):
                    temp[i] += df[old_feat[i]]

            temp[temp == 0] = 0.0099 # nie można dzielić przez zero, a najmniejsza wartość to raczej będzie 0.01 zł
            new_df[f"{new_feat}01"] = temp[0] / temp[1]
            new_df[f"{new_feat}12"] = temp[1] / temp[2]
        else:
            rec = old_feats[0]
            exp = old_feats[1]
            REC = np.array(df[rec])
            REC[REC == 0] = 0.0099
            EXP = np.array(df[exp])
            EXP[EXP == 0] = 0.0099
            temp = REC / EXP
            new_df[f"{new_feat}01"] = temp[:, 0] / temp[:, 1]
            new_df[f"{new_feat}12"] = temp[:, 1] / temp[:, 2]
            
    return pd.DataFrame(new_df), df.Y # df.iloc[:,1:]
    