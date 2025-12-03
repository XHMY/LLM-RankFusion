import scipy.io
import pandas as pd
from ranx import Run
from ranx import Qrels
from pandas import DataFrame
from ranx import compare


def get_qrels_file(RA_rel_file):
    # 使用 Pandas 的 read_csv 函数加载 CSV 文件
    df = pd.read_csv(RA_rel_file, header=None)
    df.columns = ["q_id", "unuse", "doc_id", "relevance"]

    # 使用 apply 函数创建新的 DataFrame qrels_df

    qrels_df = df.apply(lambda row: pd.Series({
        "q_id": "q_" + str(row["q_id"]),
        "doc_id": "d_" + str(row["doc_id"]),
        "score": row["relevance"]
    }), axis=1)
    # 打印 DataFrame 的前几行数据
    # print(df)
    # print(qrels_df)

    # qrels_df["score"] = qrels_df["score"].astype(int)

    qrels = Qrels.from_df(
        df=qrels_df,
        q_id_col="q_id",
        doc_id_col="doc_id",
        score_col="score",
    )
    return qrels


# 处理自己算法的RA结果文件
def get_test_run_file(RA_test_file):
    df = pd.read_csv(RA_test_file, header=None)
    df.columns = ["q_id", "doc_id", "itemrank"]

    max_rank_value = df["itemrank"].max()
    print(max_rank_value)

    run_df = df.apply(lambda row: pd.Series({
        "q_id": "q_" + str(row["q_id"]),
        "doc_id": "d_" + str(row["doc_id"]),
        "score": max_rank_value - row["itemrank"] + 1
    }), axis=1)
    # print(df)
    # print(qrels_df)

    run_df["score"] = run_df["score"].astype(float)

    run = Run.from_df(
        df=run_df,
        q_id_col="q_id",
        doc_id_col="doc_id",
        score_col="score",
    )
    return run


# 处理FLAGR库的RA结果文件

def get_csv_run_file(RA_run_file):
    # 使用 Pandas 的 read_csv 函数加载 CSV 文件
    df = pd.read_csv(RA_run_file, header=None)
    df.columns = ["q_id", "dataset", "doc_id", "itemrank", "itemscore"]
    # 使用 apply 函数创建新的 DataFrame qrels_df

    max_rank_value = df["itemrank"].max()
    print(max_rank_value)

    run_df = df.apply(lambda row: pd.Series({
        "q_id": "q_" + str(row["q_id"]),
        "doc_id": "d_" + str(row["doc_id"]),
        "score": max_rank_value - row["itemrank"] + 1
    }), axis=1)
    # 打印 DataFrame 的前几行数据
    # print(df)
    # print(qrels_df)

    run_df["score"] = run_df["score"].astype(float)

    run = Run.from_df(
        df=run_df,
        q_id_col="q_id",
        doc_id_col="doc_id",
        score_col="score",
    )
    return run


input_file = r'rank-result-testdata-RA_Copeland.csv'
input_rel_file = r'C:/Users/86137/Desktop/Tancilon_RA/FLAGR_testdata_qrels.csv'


# 评测FLAGR库的结果用这个函数转换
ranx_input_file = get_csv_run_file(input_file)

# 评测自己的方法结果用这个
# ranx_input_file = get_test_run_file(input_file)


ranx_input_rel_file = get_qrels_file(input_rel_file)

# Compare different runs and perform statistical tests
report = compare(
    qrels=ranx_input_rel_file,
    runs=[ranx_input_file],
    metrics=["recall@10", "recall@20", "recall@30",
             "recall@40", "recall@50", "ndcg@10", "ndcg@20", "ndcg@30", "ndcg@40", "ndcg@50"],
    # metrics=["map@3000"],
    max_p=0.01  # P-value threshold
)

print(report)
