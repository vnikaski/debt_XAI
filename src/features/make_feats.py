import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)

dic_feat = {'Kids': [['X1', 'X2', 'X3'], ['X64', 'X65', 'X66'], ['X169', 'X170', 'X171']],
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

def div_feat(temp, **kwargs):
    
    temp[temp == 0] = 0.0099
    
    return temp[0] / temp[1], temp[1] / temp[2]

def sub_feat(temp, **kwargs):
    
    return temp[0] - temp[1], temp[1] - temp[2]

def lin_feat(temp, **kwargs):
    
    coef = []
    for feat in temp.T:
        if np.sum(feat == 0) != 3:
            a, b = np.polyfit(feat, [1, 3, 6], 1)
        else:
            a = 0
        coef.append(a)
    
    return np.array(coef)[np.newaxis]

def three_feat(temp, **kwargs):
    
    return temp

dic_method = {'div': div_feat, 'sub': sub_feat, 'lin': lin_feat, 'old': None, 'three': three_feat}

def make_feats(df, method, prop):
    if method == 'old':
        return df
    else:
        new_df = {}
        for new_feat, old_feats in dic_feat.items():
            if new_feat != 'All_prop':
                temp = np.zeros((3, len(df)))
                for old_feat in old_feats:
                    for i in range(3):
                        temp[i] += df[old_feat[i]]
            else:
                rec = old_feats[0]
                exp = old_feats[1]
                REC = np.array(df[rec])
                REC[REC == 0] = 0.0099
                EXP = np.array(df[exp])
                EXP[EXP == 0] = 0.0099
                temp = (REC / EXP).T
                
            tab = dic_method[method](temp)
            for i, t in enumerate(tab):
                new_df[f"{new_feat}_{method}_{i}"] = t
            
        if prop:
            for col in new_df:
                if 'Allowance' in col or 'Earnings' in col:
                    new_df[col] /= REC[:, int(col[-1])]
                elif 'All_prop' not in col:
                    new_df[col] /= EXP[:, int(col[-1])]

        return pd.DataFrame(new_df)
    