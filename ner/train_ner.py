from transformers_ner import TransformersNER


def run_bert():
    cfg = {'checkpoint_dir': 'logs/xlm-roberta-base',
            'dataset': 'dataset/150',
            'transformers_model': 'xlm-roberta-base',
            'lr': 5e-6,
            'epochs': 20,
            # 'epochs': 1,#for debug
            'max_seq_length': 64}
    #for test, consider adding cache_dir='cache_dir' to the cfg
    model = TransformersNER(cfg)
    model.train()


if __name__ == '__main__':
    run_bert()
