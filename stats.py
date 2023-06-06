from scipy.stats import wilcoxon
from statistics import mean, stdev
import pandas as pd
import pingouin as pg


def get_ordered_vals(CTS_ids,healthy_ids,df,par):
    CTS_vals,healthy_vals=[],[]
    for CTS_id in CTS_ids:
        CTS_id = 'SUB0C00'+CTS_id[-2:]
        CTS_vals.append(df.at[CTS_id,par])
    for healthy_id in healthy_ids:
        healthy_id = 'SUB0000'+healthy_id[-2:]
        healthy_vals.append(df.at[healthy_id,par])

    return CTS_vals,healthy_vals

def calc_avg_std(CTS_vals,healthy_vals):
    CTS_avg,healthy_avg = mean(CTS_vals),mean(healthy_vals)
    CTS_std,healthy_std = stdev(CTS_vals),stdev(healthy_vals)
    
    return healthy_avg,healthy_std,CTS_avg,CTS_std

def calc_ICC(gt_vals,pred_vals,sub_id):
    scores = gt_vals+pred_vals
    rater = ['gt']*len(gt_vals) + ['pred']*len(pred_vals)
    d={'sub_id':sub_id+sub_id,'rater':rater,'scores':scores}
    df = pd.DataFrame(d,columns=['sub_id','rater','scores'])
    results = pg.intraclass_corr(data=df, targets='sub_id', raters='rater', ratings='scores')
    
    return results




pred = pd.read_excel(r'Morph_params.xlsx',sheet_name='pred')
sub_ids=pred['sub_id']
pred=pred.set_index('sub_id')

gt = pd.read_excel(r'Morph_params.xlsx',sheet_name='gt')
gt = gt.set_index('sub_id')

sub_match = pd.read_csv(r'Subject_Match.csv')
healthy_ids = list(sub_match['Healthy_id'])
sub_match = sub_match.set_index(['Healthy_id'])



parameters = list(pred.columns.values[1:])
CTS_ids=[]
for id in healthy_ids:
    match = sub_match.at[id,'CTS_id']
    CTS_ids.append(match)

#calc mean, std, wilcoxon test
wilc_s,wilc_p,wilc_z,CTS_means,healthy_means,CTS_stds,healthy_stds,wilc_pred_p,wilc_pred_s,ICCs,mean_errs,std_errs,wilc_s_gt,wilc_p_gt=[],[],[],[],[],[],[],[],[],[],[],[],[],[]
error_df=pd.DataFrame(columns=parameters)
for par in parameters:

    CTS_vals,healthy_vals = get_ordered_vals(CTS_ids,healthy_ids,pred,par)
    CTS_vals_gt,healthy_vals_gt = get_ordered_vals(CTS_ids,healthy_ids,pred,par)

    error_df[par]=abs(pred[par]-gt[par])
    mean_err = mean(error_df[par])
    std_err = stdev(error_df[par])
    mean_errs.append(mean_err)
    std_errs.append(std_err)

    #Compare GT vs Prediction
    wilc_pred = wilcoxon(pred[par],gt[par])
    wilc_pred_s.append(wilc_pred.statistic)
    wilc_pred_p.append(wilc_pred.pvalue)

    

    #Wilcoxon's Test for comparing CTS vs healthy
    wilc = (wilcoxon(CTS_vals,healthy_vals))
    wilc_s.append(wilc.statistic)
    wilc_p.append(wilc.pvalue)

    wilc_gt = (wilcoxon(CTS_vals_gt,healthy_vals_gt))
    wilc_s_gt.append(wilc_gt.statistic)
    wilc_p_gt.append(wilc_gt.pvalue)

    healthy_avg,healthy_std,CTS_avg,CTS_std = calc_avg_std(CTS_vals,healthy_vals)
    CTS_means.append(CTS_avg)
    healthy_means.append(healthy_avg)
    CTS_stds.append(CTS_std)
    healthy_stds.append(healthy_std)

    ICC = calc_ICC(list(pred[par]),list(gt[par]),list(sub_ids))
    ICC=ICC.set_index(['Type'])
    ICC=ICC.loc['ICC3']
    ICCs.append(ICC)
    

    

    #ICC

d = {'Parameter':parameters,'statistic':wilc_s,'p':wilc_p,'CTS Mean':CTS_means,'CTS STD':CTS_stds,'Healthy Mean':healthy_means,'Healthy STD':healthy_stds,'Pred_p':wilc_pred_p,'mean_err':mean_errs,'std_err':std_errs,'statistic_gt':wilc_s_gt,'p_gt':wilc_p_gt}
wilc_df = pd.DataFrame(d,columns=['Parameter','statistic','p'])
wilc_gt_df = pd.DataFrame(d,columns=['Parameter','statistic_gt','p_gt'])
means_df=pd.DataFrame(d,columns=['Parameter','Healthy Mean','Healthy STD','CTS Mean','CTS STD','mean_err','std_err'])
ICC_df=pd.DataFrame(ICCs)
print(ICC_df)
ICC_df['parameter']=parameters
ICC_df=ICC_df.reset_index()
ICC_df=ICC_df.set_index(['parameter'])


with pd.ExcelWriter('morph_params.xlsx',mode='a',if_sheet_exists='replace') as writer:
    means_df.to_excel(writer,sheet_name='mean_std')
    wilc_df.to_excel(writer,sheet_name='wilc_test_pred')
    wilc_gt_df.to_excel(writer,sheet_name='wilc_test_gt')
    error_df.to_excel(writer,sheet_name='error')
    ICC_df.to_excel(writer,sheet_name='ICC')
print(wilc_df)


