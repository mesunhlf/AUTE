# MODEL OPTS
def model_args(parser):
    group = parser.add_argument_group('Model', 'Arguments control Model')
    group.add_argument('--arch', default='ResNet20', type=str,
                       choices=['ResNet20', 'VggNet', 'PreAct_ResNet18', 'ResNet18', 'WideResNet34'],
                       help='model architecture')
    group.add_argument('--depth', default=20, type=int,
                       help='depth of the model')
    group.add_argument('--model-num', default=3, type=int,
                       help='number of submodels within the ensemble')
    group.add_argument('--model-file', default=None, type=str,
                       help='Path to the file that contains model checkpoints')
    group.add_argument('--gpu', default='1', type=str,
                       help='gpu id')
    group.add_argument('--seed', default=0, type=int,
                       help='random seed for torch')


# DATALOADING OPTS
def data_args(parser):
    group = parser.add_argument_group('Data', 'Arguments control Data and loading for training')
    group.add_argument('--data-dir', type=str, default='./data',
                       help='Dataset directory')
    group.add_argument('--batch-size', type=int, default=128,
                       help='batch size of the train loader')
    group.add_argument('--num-class', default=10, type=int,
                       help='number of class')

def arow_train_args(parser):
    group = parser.add_argument_group('GAL Training', 'Arguments to configure GAL training')
    group.add_argument('--gamma', default=7.0, type=float,
                       help='coefficient for coherence')


# BASE TRAINING ARGS
def base_train_args(parser):
    group = parser.add_argument_group('Base Training', 'Base arguments to configure training')
    group.add_argument('--epochs', default=120, type=int,
                       help='number of training epochs')
    group.add_argument('--lr', default=0.1, type=float,
                       help='learning rate')
    group.add_argument('--sch-intervals', nargs='*', default=[80,100], type=int,
                       help='learning scheduler milestones')
    group.add_argument('--lr-gamma', default=0.1, type=float,
                       help='learning rate decay ratio')

# ADVERSARIAL TRAINING ARGS
def adv_train_args(parser):
    group = parser.add_argument_group('Adversarial Training', 'Arguments to configure adversarial training')
    group.add_argument('--eps', default=8./255., type=float, 
                       help='perturbation budget for adversarial training')
    group.add_argument('--alpha', default=2./255., type=float,
                       help='step size for adversarial training')
    group.add_argument('--steps', default=1, type=int,
                       help='number of steps for adversarial training')

# MY TRAINING ARGS
def my_train_args(parser):
    group = parser.add_argument_group('My Training', 'Arguments to configure adversarial training')
    group.add_argument('--eps', default=8./255., type=float, 
                       help='perturbation budget for adversarial training')
    group.add_argument('--alpha', default=2./255., type=float, 
                       help='step size for adversarial training')
    group.add_argument('--steps', default=4, type=int, 
                       help='number of steps for adversarial training')
    group.add_argument('--beta', default=1.0, type=float,
                       help='number of steps for adversarial training')
    group.add_argument('--gamma', default=0.01, type=float,
                       help='number of unlearning')
    group.add_argument('--margin', default=0.01, type=float,
                       help='coefficient for coherence')
    group.add_argument('--ls', default=0.1, type=float,
                       help='coefficient for coherence')
    group.add_argument('--start-from', default='baseline', type=str, choices=['baseline', 'scratch'],
                       help='starting point of the training')



# WBOX EVALUATION ARGS
def wbox_eval_args(parser):
    group = parser.add_argument_group('White-box Evaluation', 'Arguments to configure evaluation of white-box robustness')
    group.add_argument('--subset-num', default=10000, type=int,
                       help='number of samples of the subset, will use the full test set if none')
    group.add_argument('--random-start', default=1, type=int,
                       help='number of random starts for PGD')
    group.add_argument('--steps', default=10, type=int,
                       help='number of steps for PGD')
    group.add_argument('--test-size', default=500, type=int,
                       help='number of test data')
    group.add_argument('--loss-fn', default='xent', type=str, choices=['xent', 'cw'],
                       help='which loss function to use')
    group.add_argument('--cw-conf', default=.1, type=float,
                       help='confidence for cw loss function')
    group.add_argument('--save-to-csv', action="store_true",
                       help='whether save the results to a csv file')
    group.add_argument('--overwrite', action="store_false", dest="append_out",
                       help='when saving results, whether use append mode')
    group.add_argument('--convergence-check', action="store_true", 
                       help='whether perform sanity check to make sure the attack converges')


# BBOX TRANSFER EVALUATION ARGS
def bbox_eval_args(parser):
    group = parser.add_argument_group('Black-box Evaluation', 'Arguments to configure evaluation of black-box robustness')
    group.add_argument('--folder', default='transfer_adv_examples', type=str, 
                       help='name of the folder that contains transfer adversarial examples')
    group.add_argument('--steps', default=100, type=int,
                       help='number of PGD steps for convergence check')
    group.add_argument('--which-ensemble', default='baseline', choices=['baseline', 'dverge', 'adp', 'gal'],
                       help='transfer from which ensemble')
    group.add_argument('--save-to-csv', action="store_true",
                       help='whether save the results to a csv file')
    group.add_argument('--overwrite', action="store_false", dest="append_out",
                       help='when saving results, whether use append mode')
    group.add_argument('--is-train', default=False, type=bool,
                       help='is train or not')
                       

# TRANSFERABILITY EVALUATION ARGS
def transf_eval_args(parser):
    group = parser.add_argument_group('Transferability Evaluation', 'Arguments to configure evaluation of transferablity among submodels')
    group.add_argument('--subset-num', default=1000, type=int, 
                       help='number of samples of the subset')
    group.add_argument('--random-start', default=5, type=int, 
                       help='number of random starts for PGD')
    group.add_argument('--steps', default=50, type=int, 
                       help='number of steps for PGD')
    group.add_argument('--save-to-file', action="store_true",
                       help='whether save the results to a file')


# DIVERSITY EVALUATION ARGS
def diversity_eval_args(parser):
    group = parser.add_argument_group('Diversity Evaluation', 'Arguments to configure evaluation of diversity of the ensemble')
    group.add_argument('--subset-num', default=1000, type=int, 
                       help='number of samples of the subset')
    group.add_argument('--save-to-file', action="store_true",
                       help='whether save the results to a file')
    group.add_argument('--is-train', default=False, type=bool,
                       help='is train or not')


# INPUT DIVERSE ARGS
def input_diverse_args(parser):
    group = parser.add_argument_group('Input Diverse Evaluation', 'Arguments to configure evaluation of diversity of the ensemble')
    group.add_argument('--data-dir', type=str, default='./data',
                       help='Dataset directory')
    group.add_argument('--batch-size', type=int, default=128,
                       help='batch size of the train loader')
    group.add_argument('--arch', default='ResNet', type=str, choices=['ResNet'],
                       help='model architecture')
    group.add_argument('--depth', default=20, type=int,
                       help='depth of the model')
    group.add_argument('--model-num', default=3, type=int,
                       help='number of submodels within the ensemble')
    group.add_argument('--model-file', default=None, type=str,
                       help='Path to the file that contains model checkpoints')
    group.add_argument('--gpu', default=0, type=str,
                       help='gpu id')
    group.add_argument('--seed', default=0, type=int,
                       help='random seed for torch')
    group.add_argument('--save-path', default='results/input_diverse', type=str,
                       help='random seed for torch')
    group.add_argument('--save-to-csv', action="store_true",
                       help='whether save the results to a csv file')
    group.add_argument('--distill-layer', default=[7, 13, 19], type=list,
                       help='which layer is used for distillation, only useful when distill-fixed-layer is True')
    group.add_argument('--distill-eps', default=0.07, type=float,
                       help='perturbation budget for distillation')
    group.add_argument('--distill-alpha', default=0.007, type=float,
                       help='step size for distillation')
    group.add_argument('--distill-steps', default=10, type=int,
                       help='number of steps for distillation')


# SGM ARGS
def sgm_args(parser):
    group = parser.add_argument_group('SGM Attack', 'Arguments to configure sgm attack')
    group.add_argument('--output-dir', default='transfer_adv_examples_sa', help='the path of the saved dataset')
    # group.add_argument('--epsilon', default=16, type=float, help='perturbation')
    group.add_argument('--num-steps', default=100, type=int, help='perturb number of steps')
    # group.add_argument('--step-size', default=2, type=float, help='perturb step size')
    group.add_argument('--gamma', default=0.2, type=float)
    group.add_argument('--momentum', default=0.0, type=float)
    # group.add_argument('--print_freq', default=10, type=int)
    # group.add_argument('--loss-fn', default='xent', type=str, choices=['xent', 'cw'],
    #                    help='which loss function to use')
    group.add_argument('--cw-conf', default=.1, type=float,
                       help='confidence for cw loss function')
    group.add_argument('--subset-num', default=1000, type=int,
                       help='number of samples of the subset, will use the full test set if none')
    group.add_argument('--mdi-prob', default=0.5, type=float,
                       help='probability of input translate of M-DI2-FGSM')
    group.add_argument('--random-start', default=3, type=int,
                       help='number of random starts for mPGD')


def boundary_plot_args(parser):
    group = parser.add_argument_group('Decision Region Plot', 'Arguments to configure region prediction')
    group.add_argument('--steps', default=1000, help='perturb number of steps along each direction')
    group.add_argument('--vmax', default=0.1, type=int, help='maximum distance along each direction')
    group.add_argument('--sample-num', default=1000, help='perturb number of steps along each direction')
    group.add_argument('--type', default='both', help='perturb number of steps along each direction')
