import glob
import os

import torch
from dataset import baseDataset
from pandas import pandas as pd
from tqdm import tqdm
from transformers import BertModel
from util import logger

class Calculator:
    def __init__(
        self,
        model: BertModel,
        dataset: baseDataset,
        save_file_path: str,
        config: dict,
        args: dict,
    ) -> None:
        if args.is_master:
            logger.info("Initializing Calculator...")
        self.config = config
        self.args = args
        self.file_num = args.file_num
        self.save_file_path = save_file_path

        self.model = model

        self.layer_num = model.config.num_hidden_layers
        self.head_num = model.config.num_attention_heads

        self.dataset = dataset

        self.attention_list = []

        # Data sampling
        sampler = torch.utils.data.DistributedSampler(
            self.dataset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
            drop_last=True,
        )

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    
    def calc(self) -> None:
        if self.args.is_master:
            logger.info("Start Calculating...")

        iter_bar = tqdm(self.dataloader, disable=not self.args.is_master)
        for data in iter_bar:
            s1_input_ids_len = data["s1_tok"].input_ids.squeeze(0).shape[1]
            s2_input_ids_len = data["s2_tok"].input_ids.squeeze(0).shape[1]

            if s1_input_ids_len != s2_input_ids_len:
                continue
            if self.args.model_max_length <= s1_input_ids_len:
                continue

            self.step(data)

        self.write_file()

        torch.distributed.barrier()
        self.concat_csv()

    def step(self, data) -> None:
        with torch.no_grad():
            # attentions:(Layer, batch_size, head, token, token)
            output1 = self.model(
                input_ids=data["s1_tok"].input_ids.squeeze(0).to(self.args.device),
                output_attentions=True,
                output_norms=True,
                return_dict=True,
            )
            output2 = self.model(
                input_ids=data["s2_tok"].input_ids.squeeze(0).to(self.args.device),
                output_attentions=True,
                output_norms=True,
                return_dict=True,
            )
        
        attentions1 = output1.attentions
        attentions2 = output2.attentions

        norms1 = output1.norms
        norms2 = output2.norms

        
        
        pair_diff_list = []
        for l in range(self.layer_num):
            for h in range(self.head_num):

                if self.args.mode == "attention_weights":
                    diff = self.calc_diff(
                        attentions1[l][0][0][h], attentions2[l][0][0][h]
                    )
                elif self.args.mode == "norms" or self.args.mode == "norms_abs":

                    mode = 1
                    head_norms1, head_norms2 =  norms1[l][mode][0][h], norms2[l][mode][0][h]

                    # TODO: SEP・CLS・句読点・それ以外のトークンの分類を行う
                    if self.args.mode == "norms_abs":
                        token_indices1 = data["s1_tok"].input_ids.squeeze(0).squeeze(0)
                        token_indices2 = data["s2_tok"].input_ids.squeeze(0).squeeze(0)
                        abs_head_norms1, abs_head_norms2 = self.sort_in_abs_pos_in_list(
                            head_norms1, head_norms2, token_indices1, token_indices2
                        )

                    diff = self.calc_diff(abs_head_norms1, abs_head_norms2) 

                diff = diff.cpu().numpy()
                pair_diff_list.append(diff)

        assert len(pair_diff_list) == self.layer_num * self.head_num
        self.attention_list.append(pair_diff_list)

    def calc_diff(self, attention1, attention2):
        # torchからnumpy配列への変換を削除し、PyTorchの操作に置き換えます
        sim = torch.abs(attention1 - attention2)
        diff = torch.max(sim)

        # # 行列の正規化
        # min_val = torch.min(sim)
        # max_val = torch.max(sim)
        # normalized_sim = (sim - min_val) / (max_val - min_val)
        #
        # # token * token
        # epsilon = 1e-10
        # percentage_matrix = normalized_sim / (attention1 + epsilon)

        # 行列の平均を取る
        # inner_product = torch.mean(percentage_matrix)
        # inner_product = torch.mean(sim)

        if torch.isnan(diff).any():
            raise ValueError("diff is detected nan.")
        return diff

    def sort_in_abs_pos_in_list(self,attention1, attention2, token_indices1, token_indices2):
        token_indices1 = token_indices1.cuda()
        token_indices2 = token_indices2.cuda()  
        
        sorted_indices1 = torch.argsort(token_indices1)
        sorted_indices2 = torch.argsort(token_indices2)

        sorted_attention1 = torch.index_select(attention1, 0, sorted_indices1)
        sorted_attention2 = torch.index_select(attention2, 0, sorted_indices2)
        
        return sorted_attention1, sorted_attention2

    def calc_in_absolute_position_in_queue(self, attention1, attention2, sorted_indices1:list[int], sorted_indices2:list[int]):
        sorted_indices1 = sorted_indices1.cuda()
        sorted_indices2 = sorted_indices2.cuda()

        sorted_attention1 = attention1.clone()
        sorted_attention2 = attention2.clone()

        for i in range(len(sorted_indices1)):
            sorted_attention1[i] = torch.index_select(
                sorted_attention1[i], 0, sorted_indices1
            )

        for k in range(len(sorted_indices2)):
            sorted_attention2[k] = torch.index_select(
                sorted_attention2[k], 0, sorted_indices2
            )

        sorted_attention1 = torch.index_select(sorted_attention1, 0, sorted_indices1)
        sorted_attention2 = torch.index_select(sorted_attention2, 0, sorted_indices2)

        return sorted_attention1, sorted_attention2

    def write_file(self) -> None:
        logger.info("Writing file...")
        # attention_listをcsvに書き込み
        with open(self.save_file_path + str(self.args.local_rank) + ".csv", "a") as f:
            for i in range(len(self.attention_list)):
                for j in range(len(self.attention_list[i])):
                    f.write(str(self.attention_list[i][j]))
                    f.write(",")
                f.write("\n")
        # memory clear
        self.attention_list = []
        torch.cuda.empty_cache()

    def concat_csv(self):
        if not self.args.is_master:
            return
        target_file_list = [
            self.save_file_path + str(i) + ".csv" for i in range(self.args.world_size)
        ]

        # header情報とcsvファイルの中身を追加していくリストを用意
        header_list = []
        data_list = []

        # header情報の書き込み
        for i in range(self.layer_num):
            for j in range(self.head_num):
                header_list.append(f"layer{i+1}_head{j+1}")

        # 読み込むファイルのリストを走査
        for file in target_file_list:
            tmp_df = pd.read_csv(file, names=header_list, index_col=False)
            data_list.append(tmp_df)

        # リストを全て行方向に結合
        df = pd.concat(data_list, axis=0, sort=False)

        # 書き込み
        df.to_csv(self.save_file_path + ".csv", header=True, index=False)

        # ファイルの削除
        for file in target_file_list:
            os.remove(file)
