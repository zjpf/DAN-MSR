from .bert4rec import Bert4Rec
from .mlm_mmoe import MlmMmoe


def get_model(conf):
    model_name = conf['train'].get('model_name')
    seq_len = conf['data'].get('seq_len')
    num_items = conf['data'].get('num_items')
    causal = conf['train'].get('causal', False)
    if model_name == "bert4rec":
        model = Bert4Rec(seq_length=seq_len, voc_size=num_items+2, num_hidden_layers=conf['train']['num_hidden_layers'],
                         num_attention_heads=conf['train']['num_attention_heads'], size_per_head=conf['train']['size_per_head'], causal=causal).model
    elif model_name in ["mlmMmoe", "mbStr"]:
        b_qkv = conf['train'].get('b_qkv', False)
        b_ff = conf['train'].get('b_ff', False)
        b_head = conf['train'].get('b_head', False)
        b_value = conf['train'].get('b_value', False)
        b_pe = conf['train'].get('b_pe', False)
        label_bhv = conf['train'].get('label_bhv', 'behaviors')
        model = MlmMmoe(seq_length=seq_len, voc_size=num_items+2, num_hidden_layers=conf['train']['num_hidden_layers'],
                        num_attention_heads=conf['train']['num_attention_heads'], size_per_head=conf['train']['size_per_head'],
                        n_e_sh=conf['train']['n_e_sh'], n_e_sp=conf['train']['n_e_sp'], n_mb=conf['data']['n_mb'], b_event=conf['train'].get('b_event'),
                        label_bhv=label_bhv, b_qkv=b_qkv, b_ff=b_ff, b_head=b_head, b_value=b_value, b_pe=b_pe, causal=causal, n_moe=conf['train'].get('n_moe'),
                        n_gate=conf['train'].get('n_gate'), b_cate=conf['train'].get('b_cate'), n_c=conf['data'].get('num_cate'),
                        n_attr=conf['data'].get('n_attr')).model
    return model
