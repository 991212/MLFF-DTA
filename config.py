
class hyperparameter():
    def __init__(self):
        self.seed = 231221
        self.learning_rate = 5e-4
        self.batch_size = 512
        self.epoch = 10000
        self.weight_decay = 1e-4
        self.patience = 50
        # self.drug_seq_feature = 'Morgen_fp'  #['mixed_fp','semantics']

        self.embedding_size = 128
        self.filter_num = 32

        self.mixed_fp_2_dim = 512
        self.mixed_fp_dropput = 0.0

        self.module_out_dim = self.filter_num * 3  # block_num=3
        self.attention_n_heads = 1

