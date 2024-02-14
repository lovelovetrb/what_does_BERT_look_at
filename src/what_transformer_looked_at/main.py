import argparse
import os

import yaml
from calculator import Calculator
from dataset import baseDataset
from dotenv import load_dotenv
from pyline_notify import notify
from transformers import AutoModel, AutoTokenizer
from util import init_config, init_gpu, logger

load_dotenv()

token = os.getenv("LINE_TOKEN")


def read_file_for_dir(dir_path: str):
    file_list = os.listdir(dir_path)
    file_list = [file for file in file_list if file.endswith(".txt")]
    return file_list


# @notify(token, debug=True, ploject_name="what transformer looked at")
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.visible_gpu])

    # config.yamlの読み込み
    logger.info("Loading config.yaml...")
    with open("src/what_transformer_looked_at/config.yaml") as f:
        config = yaml.safe_load(f)

    init_gpu(args)
    init_config(config)

    model_name = config["basic"]["model_name"]
    model = AutoModel.from_pretrained(model_name)
    model.to("cuda")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        config["basic"]["model_name"], use_fast=False
    )

    args.model_max_length = tokenizer.max_model_input_sizes[model_name]

    dir_path = read_file_for_dir(config["basic"]["data_path"])
    logger.info(f"target file num {len(dir_path)}")
    for idx, file_name in enumerate(dir_path):
        if args.debug and idx == 1:
            break

        logger.info(f"target file: {file_name}...")
        args.file_num = idx + 1

        data_path = config["basic"]["data_path"] + file_name
        dataset = baseDataset(
            data_path=data_path,
            model_max_length=args.model_max_length,
            tokenizer=tokenizer,
            args=args,
        )

        # file_nameから拡張子を取り除く
        save_file_name = file_name.split(".")[0]

        # configで設定されたディレクトリにモデル名を付加した場所に計算結果を保存する
        save_file_dir = config["basic"]["save_file_dir"] + config["basic"]["model_name"]

        # ディレクトリが存在しない場合，作成
        if not os.path.exists(save_file_dir):
            os.makedirs(save_file_dir, exist_ok=True)
        save_file_path = os.path.join(save_file_dir, save_file_name)

        # Attentionの計算
        calculator = Calculator(
            model=model,
            dataset=dataset,
            save_file_path=save_file_path,
            config=config,
            args=args,
        )
        calculator.calc()

        # ファイルの結合
    logger.info("calc done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--visible_gpu",
        type=int,
        nargs="*",
        default=[],
        help="visible gpu",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["attention_weights","norms","norms_abs"] 
    )
    args = parser.parse_args()
    main(args)
