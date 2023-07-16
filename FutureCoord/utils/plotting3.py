import os
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='../emulation-data')
    parser.add_argument('--output', type=str, default='../results_dalunwen2')
    args = parser.parse_args()

    index_mapping = {'agent_name': 'agent_name','serivce_1_accepts':'service_1_accepts','serivce_3_accepts':'service_3_accepts'}

    results = pd.DataFrame()
    linestyles={"WAC":(None,None),"SBSEA":(None,None),"FCFS":(None,None),"A2C":(None,None),"WAC(max_reward)":(None,None),"max_reward":(None,None),"WAC(prior_fairness)":(None,None),"prior_fairness":(None,None),"GRC":(None,None),"NIE":(None,None),"DP":(None,None)}
    palette={"WAC":"red","SBSEA":"blue","FCFS":"green","A2C":"yellow","WAC(max_reward)":"black","max_reward":"black","WAC(prior_fairness)":"red","GRC":"grey","NIE":"orange","DP":"purple","prior_fairness":"red"}
    markers={"WAC":"o","SBSEA":"^","FCFS":">","A2C":"p","WAC(max_reward)":"P","WAC(prior_fairness)":"o","GRC":"s","NIE":"d","DP":"8","max_reward":"P","prior_fairness":"o"}
    for table in range(1):
        data = pd.read_csv(Path(args.logdir) /
              'results.csv')
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
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax2=ax.twinx()
    # sns.barplot(x="agent",y="timeout",data=results)
    # sns.set(font='SimSun',font_scale=1.4)
    fig1 = sns.barplot(x="place", y="timeout", palette=palette, data=results)
    ax.set_xlabel("place", fontsize=15)
    ax.set_ylabel("timeout", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # fig1.legend(loc='upper left', ncol=2, columnspacing=0.1, labelspacing=0.1, handletextpad=0.1)
    # plt.setp(fig1.get_legend().get_texts())
    plt.ylim([40,80])
    # legend=fig1.get_legend()
    # for text in legend.texts:
    #     if text.get_text()=="max_reward":
    #         text.set_text("WAC(max_reward)")
    #     if text.get_text()=="prior_fairness":
    #         text.set_text("WAC(prior_fairness)")
    fig.tight_layout()
    sns.despine()
    fig.savefig(Path(args.output) / f'emulation-timeout222222.pdf')

    fig, ax = plt.subplots(figsize=(6, 6))
    # ax2=ax.twinx()
    # sns.barplot(x="agent",y="timeout",data=results)
    # sns.set(font='SimSun',font_scale=1.4)
    fig1 = sns.barplot(x="place", y="reliability",palette=palette,data=results)
    # fig1.legend(loc='upper left', ncol=2, columnspacing=0.1, labelspacing=0.1, handletextpad=0.1)
    # plt.setp(fig1.get_legend().get_texts())
    ax.set_xlabel("place", fontsize=15)
    ax.set_ylabel("reliability", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0,3.5])
    # legend=fig1.get_legend()
    # for text in legend.texts:
    #     if text.get_text()=="max_reward":
    #         text.set_text("WAC(max_reward)")
    #     if text.get_text()=="prior_fairness":
    #         text.set_text("WAC(prior_fairness)")
    fig.tight_layout()
    sns.despine()
    fig.savefig(Path(args.output) / f'emulation-error_times2.pdf')
