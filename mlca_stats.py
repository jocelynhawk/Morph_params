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


def calc_ICC(gt_vals,pred_vals,sub_id):
    scores = gt_vals+pred_vals
    rater = ['gt']*len(gt_vals) + ['pred']*len(pred_vals)
    d={'sub_id':sub_id+sub_id,'rater':rater,'scores':scores}
    df = pd.DataFrame(d,columns=['sub_id','rater','scores'])
    results = pg.intraclass_corr(data=df, targets='sub_id', raters='rater', ratings='scores')
    
    return results

def calc_wilc(df,CTS_ids,healthy_ids):
    params = list(df.columns.values[1:])
    df = df.reset_index()
    df = df.set_index(['sub_id'])
    wilc_perf_s,wilc_perf_p,CTS_mean,healthy_mean,CTS_std,healthy_std=[],[],[],[],[],[]
    
    for par in params:
        CTS,healthy = get_ordered_vals(CTS_ids,healthy_ids,df,par)

        wilc_perf = wilcoxon(healthy,CTS)
        wilc_perf_s.append(wilc_perf.statistic)
        wilc_perf_p.append(wilc_perf.pvalue)

        CTS_mean.append(mean(CTS))
        CTS_std.append(stdev(CTS))
        healthy_mean.append(mean(healthy))
        healthy_std.append(stdev(healthy))

    wilc_df = pd.DataFrame({'statistic':wilc_perf_s,'p':wilc_perf_p,'CTS_mean':CTS_mean,'CTS_std':CTS_std,'healthy_mean':healthy_mean,'healthy_std':healthy_std},columns=['statistic','p','CTS_mean','CTS_std','healthy_mean','healthy_std'])
    wilc_df['parameter']=params
    wilc_df = wilc_df.set_index(['parameter'])
    print(wilc_df)

    return wilc_df


def main(filename):
    CTS_means,healthy_means,CTS_stds,healthy_stds,wilc_pred_p,wilc_pred_s,ICCs,mean_errs,std_errs=[],[],[],[],[],[],[],[],[]

    pred = pd.read_excel(filename,sheet_name='pred')
    sub_ids=list(pred['sub_id'])
    pred=pred.set_index('sub_id')

    gt = pd.read_excel(filename,sheet_name='gt')
    gt = gt.set_index('sub_id')

    #Load age/gender-match subject names
    sub_match = pd.read_excel(r'Subject_Match.xlsx')
    healthy_ids = list(sub_match['Healthy_id'])
    sub_match = sub_match.set_index(['Healthy_id'])
    CTS_ids=[]
    for id in healthy_ids:
        match = sub_match.at[id,'CTS_id']
        CTS_ids.append(match)

    perf_eval_mdn = pd.read_excel('Eval_Params.xlsx',sheet_name='mdn')
    perf_eval_tcl = pd.read_excel('Eval_Params.xlsx',sheet_name='tcl')

    #Wilcoxon test for comparing CTS vs healthy performance params
    wilc_perf_mdn = calc_wilc(perf_eval_mdn,CTS_ids,healthy_ids)   
    wilc_perf_tcl = calc_wilc(perf_eval_tcl,CTS_ids,healthy_ids)

    #Wilcoxon Test for comparing CTS vs healthy morph params
    wilc_df_pred = calc_wilc(pred,CTS_ids,healthy_ids)
    wilc_df_gt = calc_wilc(gt,CTS_ids,healthy_ids)


    parameters = list(pred.columns.values[1:])
    error_df=pd.DataFrame(columns=parameters)



    for par in parameters:

        #Get parameter values in order according to age/gender match
        CTS_vals,healthy_vals = get_ordered_vals(CTS_ids,healthy_ids,pred,par)

        #calc mean absolute error of prediction calculations
        error_df[par]=abs(pred[par]-gt[par])
        mean_err = mean(error_df[par])
        std_err = stdev(error_df[par])
        mean_errs.append(mean_err)
        std_errs.append(std_err)

        #Compare GT vs Prediction
        wilc_pred = wilcoxon(pred[par],gt[par])
        wilc_pred_s.append(wilc_pred.statistic)
        wilc_pred_p.append(wilc_pred.pvalue)

        #calc mean and std of morph parameters
        CTS_avg,healthy_avg = mean(CTS_vals),mean(healthy_vals)
        CTS_std,healthy_std = stdev(CTS_vals),stdev(healthy_vals)
        CTS_means.append(CTS_avg)
        healthy_means.append(healthy_avg)
        CTS_stds.append(CTS_std)
        healthy_stds.append(healthy_std)

        #calc ICC for compaing prediction vs gt calculations
        ICC = calc_ICC(list(pred[par]),list(gt[par]),sub_ids)
        ICC=ICC.set_index(['Type'])
        ICC=ICC.loc['ICC3']
        ICCs.append(ICC)


    d = {'Parameter':parameters,'CTS Mean':CTS_means,'CTS STD':CTS_stds,'Healthy Mean':healthy_means,'Healthy STD':healthy_stds,'mean_err':mean_errs,'std_err':std_errs}
    


    means_df=pd.DataFrame(d,columns=['Parameter','Healthy Mean','Healthy STD','CTS Mean','CTS STD','mean_err','std_err'])

    ICC_df=pd.DataFrame(ICCs)
    ICC_df['parameter']=parameters
    ICC_df=ICC_df.reset_index()
    ICC_df=ICC_df.set_index(['parameter'])


    with pd.ExcelWriter(filename,mode='a',if_sheet_exists='replace') as writer:
        means_df.to_excel(writer,sheet_name='mean_std')
        wilc_df_pred.to_excel(writer,sheet_name='wilc_test_pred')
        wilc_df_gt.to_excel(writer,sheet_name='wilc_test_gt')
        error_df.to_excel(writer,sheet_name='error')
        ICC_df.to_excel(writer,sheet_name='ICC')
        wilc_perf_mdn.to_excel(writer,sheet_name='wilc_eval_mdn')
        wilc_perf_tcl.to_excel(writer,sheet_name='wilc_eval_tcl')


