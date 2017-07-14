# coding=utf-8
class system_config:
    def __init__(self):
        self.num_attributes = 8
        self.num_family_friendly = 2
        self.num_eatType = 3
        self.num_food = 7
        self.num_near = 1
        self.num_priceRange = 6
        self.num_area = 2
        self.num_customer_rating = 6
        self.num_name = 1
        self.readin_hidden_size = 64
        self.mlp_hidden_size = 512

        self.embedding_dim = 100
        self.batch_size = 32
        self.learning_rate = 0.01
        self.hidden_size = 512
        self.start = 1
        self.end = 2
        self.end_token = "@end@"
        self.null_token = "@null@"