import os
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='../results_final2')
    parser.add_argument('--output', type=str, default='../results_dalunwen2')
    args = parser.parse_args()

    index_mapping = {'agent_name': 'agent_name','serivce_1_accepts':'service_1_accepts','serivce_3_accepts':'service_3_accepts','ep_return':'reward'}

    results = pd.DataFrame()
    linestyles={"Fairness-AC":(None,None),"SBSEA":(None,None),"FCFS":(None,None),"A2C":(None,None),"max_reward":(None,None),"prior_fairness":(None,None),"GRC":(None,None),"DP":(None,None),"GRC_RS":(None,None)}
    palette={"Fairness-AC":"red","SBSEA":"blue","FCFS":"green","A2C":"yellow","max_reward":"black","GRC":"grey","DP":"purple","prior_fairness":"red","GRC_RS":"orange"}
    markers={"Fairness-AC":"o","SBSEA":"^","FCFS":">","A2C":"p","GRC":"s","DP":"8","max_reward":"P","prior_fairness":"o","GRC_RS":"d"}
    dirs = [directory for directory in os.listdir(args.logdir)]
    print(dirs)
    tables=[]
    for directory in dirs:
        for arrival_rate in os.listdir(args.logdir+"/"+directory):
            tables.append(Path(args.logdir) / directory / arrival_rate/'results' /
              'results.csv')
            if (Path(args.logdir) / directory / arrival_rate/'results' /
              'results.csv').exists():
                data = pd.read_csv(Path(args.logdir) / directory / arrival_rate/'results' /
              'results.csv')
                data['arrival_rate']=arrival_rate
                data.to_csv(Path(args.logdir) / directory / arrival_rate/'results' /
              'results.csv')
    tables = [table for table in tables if table.exists()]
    print(tables)
    for table in tables:
        data = pd.read_csv(table)
        data['ep_return']=data['ep_return']*8
        data['agent_name']=data['agent_name'].astype(str)
        # if data['agent_name'][0]=='Fairness-AC':
        #     print("找到Fairness-AC了")
        #     data['agent_name'][0]='WAC'
        # if data['agent_name'][0]=='GRC_RS':
        #     print("找到GRC_RS了")
        #     data['agent_name'][0]='NIE'
        # if data['agent_name'][0]=='max_reward':
        #     print("找到max_reward了")
        #     data['agent_name'][0:-1]='WAC(max_reward)'
        # if data['agent_name'][0]=='prior_fairness':
        #     print("找到prior_fairness了")
        #     data['agent_name'][0:-1]='WAC(prior_fairness)'
        results = pd.concat((results, data))
    print(results)
    results = results.rename(columns={**index_mapping})
    results = results.reset_index()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax2=ax.twinx()
    # sns.barplot(x="agent",y="timeout",data=results)
    # sns.set(font='SimSun',font_scale=1.4)
    fig1 = sns.lineplot(x="arrival_rate", y="jain_index", style='agent_name', hue='agent_name',palette=palette,dashes=linestyles,markers=markers,data=results)
    fig1.legend(loc='upper left', ncol=2, columnspacing=0.1, labelspacing=0.1, handletextpad=0.1)
    plt.setp(fig1.get_legend().get_texts())
    legend=fig1.get_legend()
    for text in legend.texts:
        if text.get_text()=="max_reward":
            text.set_text("WAC(max_reward)")
        if text.get_text()=="prior_fairness":
            text.set_text("WAC(prior_fairness)")
        if text.get_text()=="Fairness-AC":
            text.set_text("WAC")
        if text.get_text()=="GRC_RS":
            text.set_text("NIE")
    fig.tight_layout()
    sns.despine()
    fig.savefig(Path(args.output) / f'jain_index.pdf')
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax2=ax.twinx()
    # sns.barplot(x="agent",y="timeout",data=results)
    # sns.set(font='SimSun',font_scale=1.4)
    fig1 = sns.lineplot(x="arrival_rate", y="mean_dutil", style='agent_name',hue='agent_name',palette=palette,dashes=linestyles,markers=markers,data=results)
    fig1.legend(loc='upper left', ncol=2, columnspacing=0.1, labelspacing=0.1, handletextpad=0.1)
    plt.setp(fig1.get_legend().get_texts())
    legend=fig1.get_legend()
    for text in legend.texts:
        if text.get_text()=="max_reward":
            text.set_text("WAC(max_reward)")
        if text.get_text()=="prior_fairness":
            text.set_text("WAC(prior_fairness)")
        if text.get_text()=="Fairness-AC":
            text.set_text("WAC")
        if text.get_text()=="GRC_RS":
            text.set_text("NIE")
    fig.tight_layout()
    sns.despine()
    fig.savefig(Path(args.output) / 'mean_dutil.pdf')
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax2=ax.twinx()
    # sns.barplot(x="agent",y="timeout",data=results)
    # sns.set(font='SimSun',font_scale=1.4)
    fig1 = sns.lineplot(x="arrival_rate", y="accept_rate", style='agent_name',hue='agent_name',palette=palette,dashes=linestyles,markers=markers,data=results)
    fig1.legend(loc='upper left', ncol=2, columnspacing=0.1, labelspacing=0.1, handletextpad=0.1)
    plt.setp(fig1.get_legend().get_texts())
    legend=fig1.get_legend()
    for text in legend.texts:
        if text.get_text()=="max_reward":
            text.set_text("WAC(max_reward)")
        if text.get_text()=="prior_fairness":
            text.set_text("WAC(prior_fairness)")
        if text.get_text()=="Fairness-AC":
            text.set_text("WAC")
        if text.get_text()=="GRC_RS":
            text.set_text("NIE")
    fig.tight_layout()
    sns.despine()
    fig.savefig(Path(args.output) / 'accept_rate_fcfs.pdf')
    # sns.set_style("whitegrid")
    # fig, ax = plt.subplots(figsize=(8, 6))
    # # ax2=ax.twinx()
    # # sns.barplot(x="agent",y="timeout",data=results)
    # # sns.set(font='SimSun',font_scale=1.4)
    # fig1 = sns.lineplot(x="arrival_rate", y="service_3_accepts", style='agent_name',hue='agent_name',palette=palette,dashes=linestyles,markers=markers,data=results)
    # fig1.legend(loc='upper left', ncol=2, columnspacing=0.1, labelspacing=0.1, handletextpad=0.1)
    # plt.setp(fig1.get_legend().get_texts())
    # legend=fig1.get_legend()
    # for text in legend.texts:
    #     if text.get_text()=="max_reward":
    #         text.set_text("WAC(max_reward)")
    #     if text.get_text()=="prior_fairness":
    #         text.set_text("WAC(prior_fairness)")
    #     if text.get_text()=="Fairness-AC":
    #         text.set_text("WAC")
    #     if text.get_text()=="GRC_RS":
    #         text.set_text("NIE")
    # fig.tight_layout()
    # sns.despine()
    # fig.savefig(Path(args.output) / 'service_3_accepts.pdf')
    # fig, ax = plt.subplots(figsize=(8, 6))
    # fig1 = sns.lineplot(x="arrival_rate", y="service_3_accepts", style='agent_name', hue='agent_name',markers=markers,dashes=linestyles,data=results)
    # fig1.legend(loc='upper left', ncol=2, columnspacing=0.1, labelspacing=0.1, handletextpad=0.1)
    # plt.setp(fig1.get_legend().get_texts())
    # fig.tight_layout()
    # sns.despine()
    # fig.savefig(Path(args.output) / f'service_3_accepts.pdf')
    # sns.set_style("whitegrid")
    # fig, ax = plt.subplots(figsize=(8, 6))
    # # ax2=ax.twinx()
    # # sns.barplot(x="agent",y="timeout",data=results)
    # # sns.set(font='SimSun',font_scale=1.4)
    # fig1 = sns.lineplot(x="arrival_rate", y="accept_rate", style='agent_name', data=results)
    # fig1.legend(loc='upper left', ncol=2, columnspacing=0.1, labelspacing=0.1, handletextpad=0.1)
    # plt.setp(fig1.get_legend().get_texts())
    # fig.tight_layout()
    # sns.despine()
    # fig.savefig(Path(args.output) / f'accept_rate_fcfs.pdf')
    # sns.set_style("whitegrid")
    # fig, ax = plt.subplots(figsize=(8, 6))
    # # ax2=ax.twinx()
    # # sns.barplot(x="agent",y="timeout",data=results)
    # # sns.set(font='SimSun',font_scale=1.4)
    # fig1 = sns.lineplot(x="num_requests", y="jain_index", hue='agent', data=results)
    # fig1.legend(loc='upper left', ncol=2, columnspacing=0.1, labelspacing=0.1, handletextpad=0.1)
    # plt.setp(fig1.get_legend().get_texts())
    # fig.tight_layout()
    # sns.despine()
    # fig.savefig(Path(args.output) / f'jain_index.pdf')

    # sns.set_style("whitegrid")
    # for measure in measure_mapping.values():
    #     fig, ax = plt.subplots(figsize=(7, 6))
    #     sns.boxplot(x='Agent', y=measure, data=results, ax=ax)
    #     sns.despine()
    #     fig.savefig(Path(args.output) / f'{measure}.pdf')