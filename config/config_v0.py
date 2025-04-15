from ml_collections import ConfigDict
from config.constants import ALL_EVENT_TYPES, major_img_modalities


def check_config(config):
    for name in config.task_names:
        assert name in config.map_task_ncls, f'{name} not in map_task_ncls: \n{config.map_task_ncls}'
        assert name in config.train.task_criteria, f'{name} not in task_criteria'
        assert name in config.train.loss_weights, f'{name} not in loss_weights'

    config.map_task_ncls = {k: v for k, v in config.map_task_ncls.items() if k in config.task_names}
    config.train.task_criteria = {k: v for k, v in config.train.task_criteria.items() if k in config.task_names}
    config.train.loss_weights = {k: v for k, v in config.train.loss_weights.items() if k in config.task_names}
    config.category_names_per_task = {k: v for k, v in config.category_names_per_task.items() if k in config.task_names}
    config.survival_task_names = [name for name in config.task_names if name in config.survival_task_names_all]

    assert len(config.task_names) == len(config.map_task_ncls) == len(config.train.task_criteria) == \
           len(config.train.loss_weights), 'task_names, map_task_ncls, task_criteria, loss_weights should have same length'


def get_test_config():
    return get_default_config()


def get_default_config(args=None):
    config = ConfigDict()
    config.model = ConfigDict()
    config.data = ConfigDict()
    config.train = ConfigDict()
    config.args = ConfigDict()
    config.evaluation = ConfigDict()
    config.logger = None
    config.version = 'V2'

    config.expname = 'ColonAID'
    config.task_names = ['survival']  # must include all tasks
    config.survival_task_names_all = ['survival']
    config.survival_ncls = 12  # survival logits (years)
    config.survival_target_scheme = 'yr12'
    config.category_names_per_task = {}

    config.map_task_ncls = ConfigDict({
        'survival': config.survival_ncls,  # survival logits (years)
    })
    config.hidden_size = 512
    config.img_feature_keys = ['fmcib', 'SAMMed3D', 'universal-clipdriven', 'SAMMed2D']  # ['universal-clipdriven', 'SAMMed2D']  #
    config.txt_feature_keys = ['pulse', 'ernie']
    config.device = 'cuda'
    config.dist = False

    config.events_of_interest = ALL_EVENT_TYPES  # IMG_EVENTS  # TEXT_EVENTS  #
    config.img_mods_of_interest = major_img_modalities
    config.truncate_scheme = 'surgery'  # peri_op

    config.num_workers = 8
    config.batch_size = 16

    config.model_name = 'Please-Specify-Model-Name-In-This-Config'
    # model.name is a proxy for model_name
    config.model.name = config.model_name
    config.model.max_sequence_len = 2048
    config.model.nhead = 8
    config.model.dim_feedforward = 2048
    config.model.bias = True
    config.model.batch_first = True
    config.model.num_layers = 2
    config.model.cls_at_0 = True

    config.model.gpu_checkpoint = False

    # Event vectorizer
    config.model.dim_ct_former = 512
    config.model.dim_video_former = 512
    config.model.max_video_frames = 256
    config.model.move_attn_cpu = True

    config.model.event_vectorizer_type = 'gated_attn'  # 'average'

    # Event sequence encoder
    config.model.event_time_embd = True
    config.model.event_type_embd = True
    config.model.event_separator = False

    # Trajectory Transformer
    config.model.num_traj_cls_tokens = 1
    config.model.need_linear_back = False

    config.model.new_dropout_p = None

    # config.model.task_num_classes = (
    #     16,  # survival logits (years)
    #     5,  # T-stages T0, T1, T2, T3, T4
    #     3,  # N-stages N0, N1, N2
    # )

    config.data.is_test = False
    config.data.kfold = 11
    # config.data.n_batch_merge = 2
    config.data.traj_json_folder = '/path/to/mm_traj_per_patient'
    config.data.lab_fname = '/path/to/lab_norm.pkl'
    config.data.cancer_types = ['crc']
    # config.data.splits_fname = '/hdd/experiments/mmSurv/splits/survival_splits.json'
    config.data.splits_fname = '/path/to/split.json'
    config.data.split_seed = 0
    config.data.split_ratio = None
    config.data.split_type = None
    config.data.train_key = 'train'
    config.data.val_key = 'val'
    config.data.test_key_list = []
    config.data.ignore_no_valid_target = False
    config.data.feat_path_replace_dict = None

    config.data.enable_event_dropout = False
    config.data.event_dropout = 0.8  # each event is randomly dropped with this probability
    config.data.event_type_dropout = 0.2
    config.data.event_img_mod_dropout = 0.2
    config.data.event_after_surgery_dropout = 0.2
    config.data.event_dropout_mult = None
    config.data.whole_traj_prob = 0.25
    config.data.drop_all_img = 0.1
    config.data.drop_all_txt = 0.2
    config.data.drop_all_lab = 0.2
    config.data.only_img_prob = 0.0

    config.train.base_dir_suffix = 'v3'
    config.train.output_base_dir = '/path/to/runs'
    config.train.tb_base_dir = '/path/to/tensorboard'
    config.train.log_file = 'train.log'
    config.train.max_epoch = 100
    config.train.warmup_epoch = 5
    config.train.lr = 1e-4
    config.train.weight_decay = 5e-4
    config.train.optimizer = 'AdamW'
    config.train.scheduler = 'LinearWarmupCosineAnnealingLR'
    config.train.lr_scheduler_max_epochs = max(200, config.train.max_epoch)
    config.train.use_amp = False
    config.train.latest_ckpt = 'model_latest'
    config.train.val_online_sets = 'all'

    config.train.ckpt_latest_every = 1
    config.train.val_period = config.train.max_epoch // 5
    config.train.ckpt_period = config.train.val_period
    config.train.debug = False
    config.train.task_criteria = {
        'survival': 'NLLSurvLossSimple',
    }

    config.train.loss_weights = {
        'survival': 1.0,
    }

    check_config(config)
    return config
