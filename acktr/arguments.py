import argparse
import time

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
       '--mode', default='train', help='Test trained model or train new model, test | train'
    )
    parser.add_argument(
       '--env_name', default='Bpp-v0', type=str, help='bin packing environment name'
    )
    parser.add_argument(
       '--container_size', default=(10, 10, 10), type=int, help='container size along x, y and z axis'
    )
    parser.add_argument(
        '--enable-rotation', action='store_true', default=False, help='Whether agent can rotate boxes'
    )
    parser.add_argument(
        '--load-model', action='store_true', default=False,  help='Whether to use trained model'
    )
    parser.add_argument(
        '--load-name', default='default_cut_2.pt', 
        help='The name of trained model, you can put new trained model in it'
    )
    parser.add_argument(
        '--data-name', default='cut_2.pt',
        help='The name of testing dataset'
    )
    parser.add_argument(
        '--item-size-range', default=(2,2,2,5,5,5), type=tuple, help='the item size range, (min_width, min_length, min_height, max_width, max_length, max_height)'
    )
    parser.add_argument(
        '--use-cuda', action='store_true', default=False, help='whether to use cuda'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=False, help='whether use tensorboard to tracing trainning process'
    )
    parser.add_argument(
        '--preview', default=1, type=int, help='the item number agent knows (ignored when training)'
    )
    parser.add_argument(
        '--item-seq', default='cut1', help='item sequence generators (ignored when testing), cut1|cut2|rs|trajectory'
    )
    parser.add_argument(
        '--algorithm', default='acktr', type=str,  help='algorithm used, acktr|ppo|a2c'
    )
    parser.add_argument(
        '--use-pusnet', action='store_true', default=False,
        help='enable packing-unpacking synergistic policy and action modulation'
    )
    parser.add_argument(
        '--pusnet-no-modulation', action='store_true', default=False,
        help='disable heuristic action modulation in pusnet mode'
    )
    parser.add_argument(
        '--gamma', default=0.95, type=float,  help='discount factor for rewards (default: 0.95)'
    )
    parser.add_argument(
        '--entropy_coef', default=0.01, type=float,  help='entropy term coefficient (default: 0.01)'
    )
    parser.add_argument(
        '--value_loss_coef', default=0.5, type=float,  help='value loss coefficient (default: 0.5)'
    )
    parser.add_argument(
        '--invalid_coef', default=2, type=float,  help='invalid action possibility term coefficient'
    )
    parser.add_argument(
        '--hidden_size', default=256, type=int,  help='hidden layer cell number (default: 256)'
    )
    parser.add_argument(
        '--learning_rate', default=3e-3, type=float,  help='learning rate for a2c (default: 3e-3)'
    )
    parser.add_argument(
        '--lr', default=3e-3, type=float,  help='learning rate used by acktr/a2c wrapper (default: 3e-3)'
    )
    parser.add_argument(
        '--eps', default=1e-5, type=float,  help='RMSprop optimizer epsilon (default: 1e-5)'
    )
    parser.add_argument(
        '--alpha', default=0.99, type=float,  help='RMSprop optimizer apha (default: 0.99)'
    )
    parser.add_argument(
        '--num_processes', default=16, type=int,  help='how many training CPU processes to use (default: 16)'
    )
    parser.add_argument(
        '--device', default=0, type=int,  help='device id (default: 0)'
    )
    parser.add_argument(
        '--save_interval', default=10, type=int,  help='save interval, one save per n updates (default: 100)'
    )
    parser.add_argument(
        '--log_interval', default=10, type=int,  help='log interval, one log per n updates (default: 10)'
    )
    parser.add_argument(
        '--save_model', action='store_true', default=False,  help='whether to save training model'
    )
    parser.add_argument(
        '--cases', default=100, type=int,  help='the number of sequences used for test (default 100)'
    )
    parser.add_argument(
        '--pretrain', action='store_true', default=False,  help='load whole model'
    )
    parser.add_argument(
        '--num_steps', default=5, type=int,  help='number of forward steps in A2C (default: 5)'
    )
    parser.add_argument(
        '--enable_rotation', action='store_true', default=False,  help='whether agent can rotate box'
    )
    parser.add_argument(
        '--data_name', default='cut_1.pt', help=' the name of dataset, check data_dir for details'
    )
    parser.add_argument(
        '--load_name', default='default_cut_1.pt', help='default trained model for testing or continuing training'
    )
    parser.add_argument(
        '--load_dir', default='./pretrained_models/', help='directory to load agent logs (default: ./pretrained_models/)'
    )
    parser.add_argument(
        '--save_dir', default='./saved_models/', help='directory to save agent logs (default: ./saved_models/)'
    )
    parser.add_argument(
        '--target-total', default=50, type=int, help='the target total number of items after scaling'
    )
    parser.add_argument(
        '--seed', default=1, type=int,  help='random seed (default: 1)'
    )
    parser.add_argument(
        '--reward-alpha', default=1.0, type=float,
        help='coefficient alpha for incentive reward term r_v'
    )
    parser.add_argument(
        '--reward-beta', default=1.0, type=float,
        help='coefficient beta for incentive reward term r_sv'
    )
    parser.add_argument(
        '--reward-sigma', default=0.8, type=float,
        help='coefficient sigma for punitive reward term r_w'
    )
    parser.add_argument(
        '--reward-tau', default=0.8, type=float,
        help='coefficient tau for punitive reward term r_cw'
    )
    parser.add_argument(
        '--invalid-logit-penalty', default=1e8, type=float,
        help='large penalty used to suppress invalid logits in action modulation'
    )
    parser.add_argument(
        '--enable-online-funsearch', action='store_true', default=False,
        help='enable online heuristic evolution with optional LLM generation'
    )
    parser.add_argument(
        '--funsearch-interval', default=20, type=int,
        help='run one online funsearch cycle every N policy updates'
    )
    parser.add_argument(
        '--funsearch-sample-budget', default=128, type=int,
        help='max observation samples used in each funsearch cycle'
    )
    parser.add_argument(
        '--funsearch-candidates', default=4, type=int,
        help='number of candidate programs generated per funsearch cycle'
    )
    parser.add_argument(
        '--funsearch-topk', default=8, type=int,
        help='top-k programs considered as parents in funsearch'
    )
    parser.add_argument(
        '--llm-enable', action='store_true', default=False,
        help='enable LLM-based heuristic code generation (falls back to mutation if unavailable)'
    )
    parser.add_argument(
        '--llm-model', default='gpt-4o-mini', type=str,
        help='LLM model name used for heuristic generation'
    )
    parser.add_argument(
        '--branch-update-mode', default='alternating', type=str,
        help='branch update policy in pusnet training: alternating|pack|unpack|auto'
    )
    args = parser.parse_args()

    args.device = "cuda:" + str(args.device) if args.use_cuda else "cpu"
    args.bin_size = args.container_size
    args.pallet_size = args.container_size[0]
    args.channel = 4 # legacy hmap + size maps channel count
    args.data_type = args.item_seq
    args.test = (args.mode == 'test')
    args.use_action_modulation = (args.use_pusnet and (not args.pusnet_no_modulation))
    args.action_type_num = 2 if args.use_pusnet else 1
    if args.branch_update_mode not in ['alternating', 'pack', 'unpack', 'auto']:
        raise Exception('Unsupported branch update mode \"%s\"' % args.branch_update_mode)

    box_range = args.item_size_range
    box_size_set = []
    for i in range(box_range[0], box_range[3] + 1):
        for j in range(box_range[1], box_range[4] + 1):
            for k in range(box_range[2], box_range[5] + 1):
                box_size_set.append((i, j, k))
    args.box_size_set = box_size_set

    assert args.mode in ['train', 'test']
    if args.mode == 'train' and args.load_model:
        print('continue training model \"%s\"'%args.load_name)
    if args.mode == 'test' and args.load_model:
        print('test trained model \"%s\"'%args.load_name)
    if args.mode == 'train' and not args.load_model:
        print('train new model')
    if args.mode == 'test' and not args.load_model:
        raise Exception('no trained model chosed')
    if args.mode not in ['test', 'train']:
        raise Exception('Unknown option \'%s\''%(args.mode))
    if args.item_seq not in ['cut1', 'rs', 'cut2', 'trajectory']:
        raise Exception('Unsupported generator \'%s\''%(args.item_seq))
    print('the dataset used: ', args.data_name)
    time.sleep(0.5)
    print('the range of item size:  ', args.item_size_range)
    time.sleep(0.5)
    print('the size of bin:  ', args.bin_size)
    time.sleep(0.5)
    print('the number of known items:  ', args.preview)
    time.sleep(0.5)
    print('item sequence generator:  ', args.item_seq)
    time.sleep(0.5)
    print('enable_rotation: ', args.enable_rotation)
    print('use cuda:  ', args.use_cuda)
    print('target total items: ', args.target_total)
    print('use pusnet: ', args.use_pusnet)
    print('action modulation: ', args.use_action_modulation)
    print('reward coeffs (alpha,beta,sigma,tau): ',
          (args.reward_alpha, args.reward_beta, args.reward_sigma, args.reward_tau))
    print('online funsearch: ', args.enable_online_funsearch)
    print('llm generation enabled: ', args.llm_enable)
    print('branch update mode: ', args.branch_update_mode)
    time.sleep(0.5)
    # generate item size set
    item_set = []
    for i in range(args.item_size_range[0],args.item_size_range[3]+1):
        for j in range(args.item_size_range[1],args.item_size_range[4]+1):
            for k in range(args.item_size_range[2],args.item_size_range[5]+1):
                item_set.append((i,j,k))
    args.item_set = item_set
    print('item set: ', item_set)
    return args


 
