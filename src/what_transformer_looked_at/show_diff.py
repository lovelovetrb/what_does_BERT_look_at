import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np

# ログの設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def readFileForDir(directory_path: str):
    file_list = []
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        file_list = [
            f
            for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ]
    return file_list


def main():
    with open("src/what_transformer_looked_at/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    MODEL_NAME = config["basic"]["model_name"]
    CSV_PATH = config["basic"]["save_file_dir"]
    parts_of_speech = CSV_PATH.split("/")[-2]

    READ_DIR_PATH = CSV_PATH + MODEL_NAME + "/"
    OUTPUT_DIR_PATH = (
        config["basic"]["save_fig_dir"] + parts_of_speech + "/" + MODEL_NAME + "/"
    )

    logging.info(f"MODEL NAME: {MODEL_NAME}")
    logging.info(f"READ_DIR_PATH: {READ_DIR_PATH}")

    file_list = readFileForDir(READ_DIR_PATH)
    for i in range(len(file_list)):
        file_name_no_ext = file_list[i].replace(".csv", "")
        file_name = file_list[i]

        read_file_path = READ_DIR_PATH + file_name
        out_fig_path = OUTPUT_DIR_PATH + file_name_no_ext + "/"
        logging.info(f"OUTPUT_DIR_PATH: {out_fig_path}")

        init_dir(out_fig_path)

        logging.info(f"Reading {read_file_path} ...")
        df = pd.read_csv(read_file_path, header=0)
        # header以外のデータに対してデータフレームの型をfloatに変換
        df = df.astype(float)
        logging.info("Plotting each layer one file ...")
        show_each_layer(
            df,
            out_fig_path,
        )
        # show_each_layer_hist(
        #     df,
        #     out_fig_path,
        # )


def init_dir(out_fig_path) -> None:
    if not os.path.exists(out_fig_path):
        os.makedirs(out_fig_path)


def show_each_layer(df, out_fig_path) -> None:
    layer_num = 12
    head_num = 12
    target_head_list = []
    for i in range(layer_num):
        logging.info(f"Plotting layer {i+1} ...")
        show_df = []
        for k in range(head_num):
            index_num = i * head_num + k
            scope_df = df.iloc[1 : df.shape[0], index_num]
            show_df.append(scope_df)

            # 四分位数を計算
            q25, q50, q75 = np.percentile(scope_df, [25, 50, 75])
            iqr = q75 - q25

            tranction_list = []
            for content in scope_df:
                # matplotlibのboxplotの外れ値の計算方法と同じように外れ値を省く
                if (q25 - 1.5 * iqr) <= content <= (q75 + 1.5 * iqr):
                    tranction_list.append(content)
            # target headの抽出
            # if max(tranction_list) > 0.85:
            if q75 > 0.85:
                print(f"{i+1}-{k+1} is target head = {max(tranction_list)}")
                target_head_list.append([i+1,k+1])

        plt.subplots(figsize=(12, 6))
        # plt.boxplot(show_df) # 外れ値を表示
        plt.boxplot(show_df, sym="") # 外れ値を表示しない

        # グラフのタイトルと軸ラベルを設定
        plt.title(f"Attention Difference Layer{i+1}")
        plt.xlabel("Attention Head")
        plt.ylabel("diff")
        # グラフを保存
        plt.savefig(f"./{out_fig_path}/attention_diff_layer{i+1}.png")
        # 前のグラフをクリア
        plt.clf()

    print(target_head_list)
    print(f"target head num: {len(target_head_list)}")
    print(f"target head rate: {len(target_head_list) / (head_num * layer_num)}")
    logging.info("finished!")
    
def show_each_layer_hist(df, out_fig_path) -> None:
    layer_num = 12
    head_num = 12
    for i in range(layer_num):
        for j in range(head_num):
            logging.info(f"Plotting layer {i+1} ...")
            plt.subplots(figsize=(12, 6))
            # plt.boxplot(show_df, sym="")
            index_num = i * head_num + j
            plt.hist(df.iloc[index_num], bins=100)

            # グラフのタイトルと軸ラベルを設定
            plt.title(f"Attention Difference Layer{i+1}")
            plt.xlabel("Attention Head")
            plt.ylabel("diff")
            # グラフを保存
            plt.savefig(f"./{out_fig_path}/attention_diff_layer{i+1}_head{j+1}.png")
            # 前のグラフをクリア
            plt.clf()
    logging.info("finished!")


if __name__ == "__main__":
    main()
