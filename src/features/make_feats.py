import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)

dic_feat = {'Kids_money': [['X1', 'X2', 'X3'], ['X64', 'X65', 'X66'], ['X169', 'X170', 'X171']],
            'Kids_number': [['X5', 'X6', 'X7'], ['X68', 'X69', 'X70'], ['X173', 'X174', 'X175']],
       'Health_money': [['X225', 'X226', 'X227'], ['X8', 'X9', 'X10']],
            'Health_number': [['X229', 'X230', 'X231'], ['X12', 'X13', 'X14']],
       'ATM_money': [['X15', 'X16', 'X17']],
            'ATM_number': [['X19', 'X20', 'X21']],
       'Home_money': [['X36', 'X37', 'X38'], ['X43', 'X44', 'X45'], ['X71', 'X72', 'X73'], ['X120', 'X121', 'X122'], ['X134', 'X135', 'X136'],
                ['X141', 'X142', 'X143'], ['X183', 'X184', 'X185'], ['X232', 'X233', 'X234']],
            'Home_number': [['X40', 'X41', 'X42'], ['X47', 'X48', 'X49'], ['X75', 'X76', 'X77'], ['X124', 'X125', 'X126'], ['X138', 'X139', 'X140'],
                            ['X145', 'X146', 'X147'], ['X187', 'X188', 'X189'], ['X236', 'X237', 'X238']],
       'Food_money': [['X50', 'X51', 'X52'], ['X57', 'X58', 'X59'], ['X85', 'X86', 'X87']],
            'Food_number': [['X54', 'X55', 'X56'], ['X61', 'X62', 'X63'], ['X89', 'X90', 'X91']],
       'Other_money': [['X92', 'X93', 'X94'], ['X99', 'X100', 'X101'], ['X106', 'X107', 'X108'], ['X148', 'X149', 'X150'], ['X78', 'X79', 'X80'],
                 ['X155', 'X156', 'X157'], ['X176', 'X177', 'X178'], ['X218', 'X219', 'X220'], ['X239', 'X240', 'X241'],
                 ['X29', 'X30', 'X31']],
            'Other_number': [['X96', 'X97', 'X98'], ['X103', 'X104', 'X105'], ['X110', 'X111', 'X112'], ['X152', 'X153', 'X154'], ['X82', 'X83', 'X84'],
                             ['X159', 'X160', 'X161'], ['X180', 'X181', 'X182'], ['X222', 'X223', 'X224'], ['X243', 'X244', 'X245'],
                             ['X33', 'X34', 'X35']],
       'Transport_money': [['X113', 'X114', 'X115'], ['X127', 'X128', 'X129'], ['X190', 'X191', 'X192'], ['X197', 'X198', 'X199'],
                     ['X204', 'X205', 'X206']],
            'Transport_number': [['X117', 'X118', 'X119'], ['X131', 'X132', 'X133'], ['X194', 'X195', 'X196'], ['X201', 'X202', 'X203'],
                                 ['X208', 'X209', 'X210']],
       'Insurance_money': [['X211', 'X212', 'X213']],
            'Insurance_number': [['X215', 'X216', 'X217']],
       'Allowance_money': [['X246', 'X247', 'X248'], ['X253', 'X254', 'X255'], ['X260', 'X261', 'X262'], ['X274', 'X275', 'X276']],
            'Allowance_number': [['X250', 'X251', 'X252'], ['X267', 'X258', 'X259'], ['X264', 'X265', 'X266'], ['X278', 'X279', 'X280']],
       'Earnings_money': [['X271', 'X272', 'X273'], ['X288', 'X289', 'X290']],
            'Earnings_number': [['X275', 'X276', 'X277'], ['X292', 'X293', 'X294']],
       'All_prop_money': [['X281', 'X282', 'X283'], ['X22', 'X23', 'X24']],
            'All_prop_number': [['X285', 'X286', 'X287'], ['X26', 'X27', 'X28']]
      }

dic_limited_feats = {
    'sub': ['Other', 'ATM', 'Food', 'Earnings'],
    'div': ['Health', 'ATM', 'Earnings', 'Transport'],
    'lin': ['Other', 'Food', 'All_prop', 'Earnings'],
    'three': ['ATM']
}

def income_stability(a,b,c): #If 1 month 3 months and 6 months are not deflecting in more than 45%
    if 3*a>0.55*b and 3*a<1.45*b and 2*b>0.55*c and 2*b<1.45*c:
        return 0
    else:
        return 1

def salary_stability(a,b,c): #If 1 month 3 months and 6 months are not deflecting in more than 45%
    if 3*a>0.55*b and 3*a<1.45*b and 2*b>0.55*c and 2*b<1.45*c:
        return 0
    else:
        return 1

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
        #new_df["Income_Unstability"] = df.apply(lambda x: income_stability(x["X281"], x["X282"],x["X283"]), axis = 1)
        #new_df["Salary_Unstability"] = df.apply(lambda x: salary_stability(x["X288"], x["X289"],x["X290"]), axis = 1)
        return pd.DataFrame(new_df)

def make_limited_feats(df):
    new_df = {}
    for type, categories in dic_limited_feats.items():
        for category in categories:

            if category != 'All_prop':
                for old_feats in dic_feat[category]:
                    temp = np.zeros((3, len(df)))
                    for i in range(3):
                        temp[i] += df[old_feats[i]]
            else:
                old_feats=dic_feat[category]
                rec = old_feats[0]
                exp = old_feats[1]
                REC = np.array(df[rec])
                REC[REC == 0] = 0.0099
                EXP = np.array(df[exp])
                EXP[EXP == 0] = 0.0099
                temp = (REC / EXP).T

            tab = dic_method[type](temp)
            for i, t in enumerate(tab):
                new_df[f"{category}_{type}_{i}"] = t
    return pd.DataFrame(new_df)
