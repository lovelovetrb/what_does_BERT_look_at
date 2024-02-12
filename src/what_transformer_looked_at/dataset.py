from torch.utils.data import Dataset
from transformers import BertJapaneseTokenizer

from util import logger


class baseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: BertJapaneseTokenizer,
        model_max_length: int,
        args,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()

        self.model_max_length = model_max_length
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id

        self.data_path = data_path
        with open(data_path, encoding=encoding) as f:
            self.data = f.readlines()

        self.data = [line.strip() for line in self.data]
        if args.is_master:
            self.sanity_check(self.data)
        self.data = [self.data[i : i + 2] for i in range(0, len(self.data), 2)]

    # ここで取り出すデータを指定している
    def __getitem__(self, index: int):
        s1, s2 = self.data[index]
        s1_tok = self.tokenizer.encode_plus(
            s1,
            return_tensors="pt",
        )
        s2_tok = self.tokenizer.encode_plus(
            s2,
            return_tensors="pt",
        )
        return {
            "s1_tok": s1_tok,
            "s2_tok": s2_tok,
            "s1": s1,
            "s2": s2,
        }

    def __len__(self) -> int:
        return len(self.data)

    def sanity_check(self, lines: list[str]):
        if len(lines) % 2 != 0:
            logger.error(f"EVEN ERROR: {self.data_path}")
            raise ValueError("The number of lines is not even.")
        if len(lines) == 0:
            logger.error(f"EMPTY ERROR: {self.data_path}")
            raise ValueError("The number of lines is 0.")
        if len(lines) % 2 != 1 and lines[-1] == "":
            logger.error(f"VALIDATE ERROR: {self.data_path}")
            raise ValueError("The number of validate lines is not even.")
        logger.info(f"Sanity check done! {self.data_path}")
