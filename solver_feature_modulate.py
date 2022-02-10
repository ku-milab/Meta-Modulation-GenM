import torch, os
from os.path import join
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models import *
from process_data.data_loader import *
from util import *
# from solver import Baselearner


class GenM():
    def __init__(self, flags):
        self.set_path(flags)
        self.setup_data(flags)
        self.build_models(flags)
        # set loss function
        self.loss_fn = CosineLoss(flags)
        # self.loss_fn = nn.CrossEntropyLoss()

    def set_path(self, flags):
        self.model_path = os.path.join(flags.root_path, flags.model_path, flags.exp, str(flags.fold))
        self.log_path = os.path.join(flags.root_path, flags.log_path, flags.exp, str(flags.fold))
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
        self.tb = SummaryWriter(log_dir=self.log_path)


    def setup_data(self, flags):
        self.dataloader = ABIDE_loader()
        self.base_train, self.intra_val, self.intra_test, self.episodes_ASD, self.episodes_TC = self.dataloader.setup_source(flags.root_path, flags.fc_type, flags.roi_type, flags.sources, flags.fold, flags.batch_size)
        self.inter_test = self.dataloader.setup_unseen(flags.root_path, flags.fc_type, flags.roi_type, flags.unseens)

    def build_models(self, flags):
        # base learning model
        self.feat_nets = []
        for i in range(len(flags.sources)):
            self.feat_nets.append(feature_network(flags.input_dim, flags.hidden_dim1, flags.feature_activation).to(flags.device))
        self.task_net = task_network(flags.hidden_dim1, flags.class_num).to(flags.device)

        # meta-learning
        self.feat_meta = feature_network(flags.input_dim, flags.hidden_dim1, flags.feature_activation).to(flags.device)
        self.modulation_net = modulation_netowrk(flags.input_dim, flags.hidden_dim1, flags.modulation_feature_activation).to(flags.device)

        # final model
        # self.feat_gen = feature_network(flags.input_dim, flags.hidden_dim1).to(flags.device)

        # set optimizer
        self.optim_feat_nets = []
        for i in range(len(flags.sources)):
            self.optim_feat_nets.append(torch.optim.Adam(self.feat_nets[i].parameters(), lr=flags.lr))
        self.optim_task_net = torch.optim.Adam(self.task_net.parameters(), lr=flags.lr)
        self.optim_feat_meta = torch.optim.Adam(self.feat_meta.parameters(), lr=flags.lr)
        self.optim_modulation_net = torch.optim.Adam(self.modulation_net.parameters(), lr=flags.lr)
        # self.optim_feat_gen = torch.optim.Adam(self.feat_gen.parameters(), lr=flags.lr)

        # # set scheduler
        # if flags.optim_scheduler == True:
        #     self.sch_optim_feat_nets = []
        #     for i in range(len(flags.sources)):
        #         self.sch_optim_feat_nets.append(torch.optim.lr_scheduler.LambdaLR(optimizer=self.optim_feat_nets[i],
        #                                                                           lr_lambda=lambda epoch: 0.95 ** epoch,
        #                                                                           last_epoch=-1, verbose=False))
        #     self.sch_optim_task_net = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optim_task_net,
        #                                                                 lr_lambda=lambda epoch: 0.95 ** epoch,
        #                                                                 last_epoch=-1))
        #     self.sch_optim_feat_meta = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optim_feat_meta,
        #                                                                           lr_lambda=lambda epoch: 0.95 ** epoch,
        #                                                                           last_epoch=-1, verbose=False))
        #     self.sch_optim_modulation_net = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optim_modulation_net,
        #                                                                           lr_lambda=lambda epoch: 0.95 ** epoch,
        #                                                                           last_epoch=-1, verbose=False))
            # self.sch_optim_feat_gen = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optim_feat_nets[i],
            #                                                                                   lr_lambda=lambda epoch: 0.95 ** epoch,
            #                                                                                   last_epoch=-1, verbose=False))

        # init weights
        if flags.weights_init == 'He':
            print('He init')
            for i in range(len(flags.sources)):
                self.feat_nets[i].apply(He_weights_init)
            self.task_net.apply(He_weights_init)
            self.feat_meta.apply(He_weights_init)
            self.modulation_net.apply(He_weights_init)
            # self.feat_gen.apply(He_weights_init)
        elif flags.weights_init == 'xavier':
            print('xavier init')
            for i in range(len(flags.sources)):
                self.feat_nets[i].apply(xavier_weights_init)
            self.task_net.apply(xavier_weights_init)
            self.feat_meta.apply(xavier_weights_init)
            self.modulation_net.apply(xavier_weights_init)
            # self.feat_gen.apply(xavier_weights_init)

        # count trainable params
        print('feature network: ', count_trainable_param(self.feat_meta), '\n',
              'task network: ', count_trainable_param(self.task_net), '\n',
              'modulation network: ', count_trainable_param(self.modulation_net), '\n')

    def meta_learning(self, flags):
        # start writing log
        log_train = join(self.log_path, flags.meta_learning_log)
        log_csv(log_train, 'step,base_losses,meta_train_loss,meta_test_loss,meta_train_acc,meta_test_acc,eval_loss,eval_acc,'
                           'auc,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           'acc,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           'sen,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           'spec,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           ' ,intra_acc,inter_acc'.format(*flags.sources,*flags.unseens,*flags.sources,*flags.unseens,*flags.sources,*flags.unseens,*flags.sources,*flags.unseens))

        # set data iterator
        base_iter, base_iter_len = [], []
        # meta_data_flags = [0 for _ in range(len(flags.sources))]
        for i in range(len(flags.sources)):
            base_iter.append(iter(self.base_train[i]))
            base_iter_len.append(len(base_iter[i]))
            print('train data{} iter length: '.format(i), len(base_iter[i]))

        best_meta_test_loss = 0
        best_meta_eval_loss = 0
        # best_meta_eval_acc = 0
        best_intra_acc, best_inter_acc = 0, 0

        base_iter_flag = 0
        for i in range(flags.meta_learning_steps):
            # Base model learning stage
            base_losses = [0 for _ in range(len(flags.sources))]
            # base_accs = [0 for _ in range(len(flags.sources))]
            for j in range(flags.base_steps):
                for k in range(len(flags.sources)):
                    # Check dataload sequence
                    if ((base_iter_flag+1) % base_iter_len[k])==0:
                        base_iter[k] = iter(self.base_train[k])

                    self.feat_nets[k].train(), self.task_net.train() # set train mode
                    x, y = base_iter[k].__next__()
                    x, y = x.type(torch.FloatTensor).to(flags.device), y.type(torch.LongTensor).to(flags.device)

                    self.optim_feat_nets[k].zero_grad()
                    self.optim_task_net.zero_grad()

                    features = self.feat_nets[k](x)
                    logits = self.task_net(features)
                    loss = self.loss_fn(logits, y)
                    loss.backward()

                    self.optim_feat_nets[k].step()
                    self.optim_task_net.step()

                    # if flags.optim_scheduler == True:
                    #     self.sch_optim_feat_nets[k].step()
                    #     self.sch_optim_task_net.step()

                    # check
                    # prob = F.softmax(logits, dim=1)
                    # corrects = (prob.max(1)[1].view(y.size()).data == y.data).sum().item()
                    # base_accs[k] = corrects / y.size()[0]
                    base_losses[k] = loss.item()
                base_iter_flag += 1
            base_loss = sum(base_losses) / len(flags.sources)


            # meta-learning
            self.task_net.eval()
            for j in range(flags.episode_iter_steps):

                # sample episode-sites
                # meta_idx = np.random.choice(len(flags.sources), 2, replace=False)  # p = [0.4, 0.4, 0.1, 0.1]
                meta_idx = torch.multinomial(torch.arange(len(flags.sources), dtype=torch.float), 2)
                self.feat_meta.load_state_dict(self.feat_nets[meta_idx[0]].state_dict())

                # meta-train
                self.feat_meta.train(), self.modulation_net.eval()
                asd_set, asd_label = self.episodes_ASD[meta_idx[0]]
                tc_set, tc_label = self.episodes_TC[meta_idx[0]]

                for k in range(flags.meta_train_steps):
                    # sample shots
                    asd_idx = torch.multinomial(torch.arange(len(asd_set), dtype=torch.float), flags.shot)
                    tc_idx = torch.multinomial(torch.arange(len(tc_set), dtype=torch.float), flags.shot)

                    x = torch.cat((asd_set[asd_idx], tc_set[tc_idx]), 0)
                    y = torch.cat((asd_label[asd_idx], tc_label[tc_idx]), 0)
                    x, y = x.type(torch.FloatTensor).to(flags.device), y.type(torch.LongTensor).to(flags.device)

                    self.optim_feat_meta.zero_grad()
                    self.optim_modulation_net.zero_grad()
                    self.optim_task_net.zero_grad()

                    features = self.feat_meta(x)
                    modulation_features = self.modulation_net(x)
                    if flags.modulation_method == 'dot_pro':
                        modulated_features = dot_product()(features, modulation_features)
                    elif flags.modulation_method == 'self_att':
                        modulated_features, attn = ScaledDotProductAttention()(modulation_features, features, features)
                    # elif flags.modulation_method == 'multihead_att':
                    #     modulated_features = MultiHeadAttention()()
                    else:
                        print('Not defined modulation method')
                        return
                    logits = self.task_net(modulated_features)
                    loss_meta_train = self.loss_fn(logits, y)
                    loss_meta_train.backward()
                    self.optim_feat_meta.step()

                    # if flags.optim_scheduler == True:
                    #     self.sch_optim_feat_meta.step()

                    # check
                    prob = F.softmax(logits, dim=1)
                    corrects = (prob.max(1)[1].view(y.size()).data == y.data).sum().item()
                    acc_meta_train = corrects / y.size()[0]

                # meta-test
                self.feat_meta.eval(), self.modulation_net.train()
                asd_set, asd_label = self.episodes_ASD[meta_idx[1]]
                tc_set, tc_label = self.episodes_TC[meta_idx[1]]

                for k in range(flags.meta_test_steps):
                    # sample shots
                    asd_idx = torch.multinomial(torch.arange(len(asd_set), dtype=torch.float), flags.shot)
                    tc_idx = torch.multinomial(torch.arange(len(tc_set), dtype=torch.float), flags.shot)

                    x = torch.cat((asd_set[asd_idx], tc_set[tc_idx]), 0)
                    y = torch.cat((asd_label[asd_idx], tc_label[tc_idx]), 0)
                    x, y = x.type(torch.FloatTensor).to(flags.device), y.type(torch.LongTensor).to(flags.device)

                    self.optim_feat_meta.zero_grad()
                    self.optim_modulation_net.zero_grad()
                    self.optim_task_net.zero_grad()

                    features = self.feat_meta(x)
                    modulation_features = self.modulation_net(x)
                    if flags.modulation_method == 'dot_pro':
                        modulated_features = dot_product()(features, modulation_features)
                    elif flags.modulation_method == 'self_att':
                        modulated_features, attn = ScaledDotProductAttention()(modulation_features, features, features)
                        # elif flags.modulation_method == 'multihead_att':
                        #     modulated_features = MultiHeadAttention()()
                    else:
                        print('Not defined modulation method')
                        return
                    logits = self.task_net(modulated_features)
                    loss_meta_test = self.loss_fn(logits, y)
                    loss_meta_test.backward()
                    self.optim_modulation_net.step()

                    # if flags.optim_scheduler == True:
                    #     self.sch_optim_modulation_net.step()

                    # check
                    prob = F.softmax(logits, dim=1)
                    corrects = (prob.max(1)[1].view(y.size()).data == y.data).sum().item()
                    acc_meta_test = corrects / y.size()[0]

            ############################################################################################################
            # check validation
            meta_eval_loss, [meta_eval_auc, meta_eval_acc, _, _] = self.eval(flags, self.intra_val, self.feat_meta, self.task_net, self.modulation_net)

            # check intra-test
            # intra_losses = [0 for _ in range(len(flags.sources))]
            intra_aucs = [0 for _ in range(len(flags.sources))]
            intra_accs = [0 for _ in range(len(flags.sources))]
            intra_sens = [0 for _ in range(len(flags.sources))]
            intra_specs = [0 for _ in range(len(flags.sources))]
            for j in range(len(flags.sources)):
                _, [intra_aucs[j], intra_accs[j], intra_sens[j], intra_specs[j]] = self.eval(flags, self.intra_test[j], self.feat_meta, self.task_net, self.modulation_net)
            # intra_loss = sum(intra_losses) / len(flags.sources)
            # intra_auc = sum(intra_aucs) / len(flags.sources)
            intra_acc = sum(intra_accs) / len(flags.sources)
            # intra_sen = sum(intra_sens) / len(flags.sources)
            # intra_spec = sum(intra_specs) / len(flags.sources)

            # check inter-test
            # inter_losses = [0 for _ in range(len(flags.unseens))]
            inter_aucs = [0 for _ in range(len(flags.unseens))]
            inter_accs = [0 for _ in range(len(flags.unseens))]
            inter_sens = [0 for _ in range(len(flags.unseens))]
            inter_specs = [0 for _ in range(len(flags.unseens))]
            for j in range(len(flags.unseens)):
                _, [inter_aucs[j], inter_accs[j], inter_sens[j], inter_specs[j]] = self.eval(flags, self.inter_test[j],self.feat_meta,self.task_net,self.modulation_net)
            # inter_loss = sum(inter_losses) / len(flags.unseens)
            # inter_auc = sum(inter_aucs) / len(flags.unseens)
            inter_acc = sum(inter_accs) / len(flags.unseens)
            # inter_sen = sum(inter_sens) / len(flags.unseens)
            # inter_spec = sum(inter_specs) / len(flags.unseens)

            print(
                'step[{}] base loss:{:.4f} |meta train loss:{:.4f} |meta test loss:{:.4f} |meta train acc:{:.4f} |meta test acc:{:.4f} |meta eval loss:{:.4f} |meta eval acc:{:.4f}'.format(
                    i, base_loss, loss_meta_train.item(), loss_meta_test.item(), acc_meta_train, acc_meta_test,
                    meta_eval_loss, meta_eval_acc))
            # writing log and tensorboard
            log_csv(log_train, '{},{},{},{},{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{}'.format(i, base_loss, loss_meta_train.item(), loss_meta_test.item(),
                                              acc_meta_train, acc_meta_test, meta_eval_loss, meta_eval_acc,
                                              ' ',*intra_aucs, *inter_aucs, ' ', *intra_accs, *inter_accs,
                                              ' ', *intra_sens, *inter_sens, ' ', *intra_specs, *inter_specs,
                                              ' ',intra_acc, inter_acc))
            self.tb.add_scalars('loss', {'base loss': base_loss,
                                         'meta train loss': loss_meta_train.item(),
                                         'meta test loss': loss_meta_test.item(),
                                         'meta eval loss': meta_eval_loss
                                         }, i)
            self.tb.add_scalars('acc', {'meta train acc': acc_meta_train,
                                        'meta test acc': acc_meta_test,
                                        'meta eval acc': meta_eval_acc,
                                        'intra acc': intra_acc,
                                        'inter acc': inter_acc
                                        }, i)

            # save checkpoint
            # check meta-test loss
            if not best_meta_eval_loss or meta_eval_loss < best_meta_eval_loss:
                torch.save({
                    'feat_net':self.feat_meta.state_dict(),
                    'task_net':self.task_net.state_dict(),
                    'modulation_net':self.modulation_net.state_dict(),
                    'optim_feat_net':self.optim_feat_meta.state_dict(),
                    'optim_task_net':self.optim_task_net.state_dict(),
                    'optim_modulation_net':self.optim_modulation_net.state_dict()
                }, join(self.model_path, 'meta_eval_loss_checkpoint.tar'))
                best_meta_eval_loss = meta_eval_loss
            if not best_meta_test_loss or loss_meta_test.item() < best_meta_test_loss:
                torch.save({
                    'feat_net':self.feat_meta.state_dict(),
                    'task_net':self.task_net.state_dict(),
                    'modulation_net':self.modulation_net.state_dict(),
                    'optim_feat_net':self.optim_feat_meta.state_dict(),
                    'optim_task_net':self.optim_task_net.state_dict(),
                    'optim_modulation_net':self.optim_modulation_net.state_dict()
                }, join(self.model_path, 'meta_test_loss_checkpoint.tar'))
                best_meta_test_loss = loss_meta_test.item()
            if not best_intra_acc or best_intra_acc < intra_acc:
                torch.save({
                    'feat_net': self.feat_meta.state_dict(),
                    'task_net': self.task_net.state_dict(),
                    'modulation_net': self.modulation_net.state_dict(),
                    'optim_feat_net': self.optim_feat_meta.state_dict(),
                    'optim_task_net': self.optim_task_net.state_dict(),
                    'optim_modulation_net': self.optim_modulation_net.state_dict()
                }, join(self.model_path, 'meta_intra_acc_checkpoint.tar'))
                best_intra_acc = intra_acc
            if not best_inter_acc or best_inter_acc < inter_acc:
                torch.save({
                    'feat_net': self.feat_meta.state_dict(),
                    'task_net': self.task_net.state_dict(),
                    'modulation_net': self.modulation_net.state_dict(),
                    'optim_feat_net': self.optim_feat_meta.state_dict(),
                    'optim_task_net': self.optim_task_net.state_dict(),
                    'optim_modulation_net': self.optim_modulation_net.state_dict()
                }, join(self.model_path, 'meta_inter_acc_checkpoint.tar'))
                best_inter_acc = inter_acc

            if loss_meta_test.item() == 0.0:
                return


    def GenM_train(self, flags):
        # start writing log
        log_train = join(self.log_path, flags.GenM_train_log)
        log_csv(log_train,
                'step,train_loss,eval_loss,eval_acc,'
                'auc,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'acc,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'sen,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'spec,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'feat_auc,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'feat_acc,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'feat_sen,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'feat_spec,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                ' ,intra_acc,inter_acc'.format(*flags.sources, *flags.unseens, *flags.sources, *flags.unseens,
                                             *flags.sources, *flags.unseens, *flags.sources, *flags.unseens,
                                             *flags.sources, *flags.unseens, *flags.sources, *flags.unseens,
                                             *flags.sources, *flags.unseens, *flags.sources, *flags.unseens
                                             ))

        # set data iterator
        base_iter, base_iter_len = [], []
        # meta_data_flags = [0 for _ in range(len(flags.sources))]
        for i in range(len(flags.sources)):
            base_iter.append(iter(self.base_train[i]))
            base_iter_len.append(len(base_iter[i]))
            # print('train data{} iter length: '.format(i), len(base_iter[i]))

        # set criteria
        best_train_loss = 0
        best_eval_loss = 0
        # best_eval_acc = 0
        best_intra_acc, best_inter_acc = 0, 0

        # load checkpoint
        if flags.checkpoint:
            checkpoint = torch.load(join(self.model_path,flags.checkpoint))
            self.task_net.load_state_dict(checkpoint['task_net'])
            self.optim_task_net.load_state_dict(checkpoint['optim_task_net'])
            self.modulation_net.load_state_dict(checkpoint['modulation_net'])
            self.optim_modulation_net.load_state_dict(checkpoint['optim_modulation_net'])

        # init feature network
        if flags.weights_init == 'He':
            self.feat_meta.apply(He_weights_init)
        elif flags.weights_init == 'xavier':
            self.feat_meta.apply(xavier_weights_init)

        for i in range(flags.gen_train_steps):
            # train GenM
            self.feat_meta.train(), self.task_net.eval(), self.modulation_net.eval()
            self.optim_feat_meta.zero_grad()
            self.optim_task_net.zero_grad()
            self.optim_modulation_net.zero_grad()

            losses = 0
            for j in range(len(flags.sources)):
                # Check dataload sequence
                if (i + 1) % (base_iter_len[j]) == 0:
                    base_iter[j] = iter(self.base_train[j])

                x, y = base_iter[j].__next__()
                x, y = x.type(torch.FloatTensor).to(flags.device), y.type(torch.LongTensor).to(flags.device)

                features = self.feat_meta(x)
                # logits_feat = self.task_net(features)
                modulation_features = self.modulation_net(x)
                if flags.modulation_method == 'dot_pro':
                    modulated_features = dot_product()(features, modulation_features)
                elif flags.modulation_method == 'self_att':
                    modulated_features, attn = ScaledDotProductAttention()(modulation_features, features, features)
                    # elif flags.modulation_method == 'multihead_att':
                    #     modulated_features = MultiHeadAttention()()
                else:
                    print('Not defined modulation method')
                    return
                # modulated_features = features * modulation_features
                logits_mod = self.task_net(modulated_features)
                loss = self.loss_fn(logits_mod, y)
                losses += loss
            train_loss = losses/len(flags.sources)
            train_loss.backward()
            self.optim_feat_meta.step()
            # if flags.optim_scheduler == True:
            #     self.sch_optim_feat_meta.step()

                # check
                # prob = F.softmax(logits, dim=1)
                # corrects = (prob.max(1)[1].view(y.size()).data == y.data).sum().item()
                # base_accs[k] = corrects / y.size()[0]

            ############################################################################################################
            # check validation
            eval_loss, [eval_auc, eval_acc, _, _] = self.eval(flags, self.intra_val, self.feat_meta,
                                                                             self.task_net, self.modulation_net)

            # check intra-test
            # intra_losses = [0 for _ in range(len(flags.sources))]
            intra_aucs = [0 for _ in range(len(flags.sources))]
            intra_accs = [0 for _ in range(len(flags.sources))]
            intra_sens = [0 for _ in range(len(flags.sources))]
            intra_specs = [0 for _ in range(len(flags.sources))]
            intra_feat_aucs = [0 for _ in range(len(flags.sources))]
            intra_feat_accs = [0 for _ in range(len(flags.sources))]
            intra_feat_sens = [0 for _ in range(len(flags.sources))]
            intra_feat_specs = [0 for _ in range(len(flags.sources))]
            for j in range(len(flags.sources)):
                _, [intra_aucs[j], intra_accs[j], intra_sens[j], intra_specs[j]] = self.eval(flags, self.intra_test[j],
                                                                                             self.feat_meta,
                                                                                             self.task_net,
                                                                                             self.modulation_net)
                _, [intra_feat_aucs[j], intra_feat_accs[j], intra_feat_sens[j],
                    intra_feat_specs[j]] = self.eval_feature(
                    flags, self.intra_test[j],
                    self.feat_meta,
                    self.task_net)
            # intra_loss = sum(intra_losses) / len(flags.sources)
            # intra_auc = sum(intra_aucs) / len(flags.sources)
            intra_acc = sum(intra_accs) / len(flags.sources)
            # intra_sen = sum(intra_sens) / len(flags.sources)
            # intra_spec = sum(intra_specs) / len(flags.sources)

            # check inter-test
            # inter_losses = [0 for _ in range(len(flags.unseens))]
            inter_aucs = [0 for _ in range(len(flags.unseens))]
            inter_accs = [0 for _ in range(len(flags.unseens))]
            inter_sens = [0 for _ in range(len(flags.unseens))]
            inter_specs = [0 for _ in range(len(flags.unseens))]
            inter_feat_aucs = [0 for _ in range(len(flags.unseens))]
            inter_feat_accs = [0 for _ in range(len(flags.unseens))]
            inter_feat_sens = [0 for _ in range(len(flags.unseens))]
            inter_feat_specs = [0 for _ in range(len(flags.unseens))]
            for j in range(len(flags.unseens)):
                _, [inter_aucs[j], inter_accs[j], inter_sens[j], inter_specs[j]] = self.eval(flags, self.inter_test[j],
                                                                                             self.feat_meta,
                                                                                             self.task_net,
                                                                                            self.modulation_net)
                _, [inter_feat_aucs[j], inter_feat_accs[j], inter_feat_sens[j],
                    inter_feat_specs[j]] = self.eval_feature(
                    flags, self.inter_test[j], self.feat_meta, self.task_net)
            # inter_loss = sum(inter_losses) / len(flags.unseens)
            # inter_auc = sum(inter_aucs) / len(flags.unseens)
            inter_acc = sum(inter_accs) / len(flags.unseens)
            # inter_sen = sum(inter_sens) / len(flags.unseens)
            # inter_spec = sum(inter_specs) / len(flags.unseens)

            print('step[{}] train loss:{:.4f} |eval loss:{:.4f} |eval acc:{:.4f} |Intra acc:{:.4f} |Inter acc:{:.4f}'.format(i, train_loss.item(), eval_loss, eval_acc, intra_acc, inter_acc))
            # writing log and tensorboard
            log_csv(log_train, '{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                               '{},{},{}'.format(i, train_loss.item(), eval_loss, eval_acc,
                                              ' ', *intra_aucs, *inter_aucs, ' ', *intra_accs, *inter_accs,
                                              ' ', *intra_sens, *inter_sens, ' ', *intra_specs, *inter_specs,
                                              ' ', *intra_feat_aucs, *inter_feat_aucs, ' ', *intra_feat_accs, *inter_feat_accs,
                                              ' ', *intra_feat_sens, *inter_feat_sens, ' ', *intra_feat_specs, *inter_feat_specs,
                                              ' ', intra_acc, inter_acc))
            self.tb.add_scalars('GenM loss', {'train loss': train_loss.item(),'eval loss': eval_loss}, i)
            self.tb.add_scalars('GenM acc', {'eval acc': eval_acc,'intra acc': intra_acc,'inter acc': inter_acc}, i)

            # save checkpoint
            # check meta-test loss
            if not best_eval_loss or eval_loss < best_eval_loss:
                torch.save({
                    'feat_net': self.feat_meta.state_dict(),
                    'task_net': self.task_net.state_dict(),
                    'modulation_net': self.modulation_net.state_dict(),
                    'optim_feat_net': self.optim_feat_meta.state_dict(),
                    'optim_task_net': self.optim_task_net.state_dict(),
                    'optim_modulation_net': self.optim_modulation_net.state_dict()
                }, join(self.model_path, 'GenM_eval_loss_checkpoint.tar'))
                best_eval_loss = eval_loss
            if not best_train_loss or train_loss.item() < best_train_loss:
                torch.save({
                    'feat_net': self.feat_meta.state_dict(),
                    'task_net': self.task_net.state_dict(),
                    'modulation_net': self.modulation_net.state_dict(),
                    'optim_feat_net': self.optim_feat_meta.state_dict(),
                    'optim_task_net': self.optim_task_net.state_dict(),
                    'optim_modulation_net': self.optim_modulation_net.state_dict()
                }, join(self.model_path, 'GenM_train_loss_checkpoint.tar'))
                best_train_loss = train_loss.item()
            if not best_intra_acc or best_intra_acc < intra_acc:
                torch.save({
                    'feat_net': self.feat_meta.state_dict(),
                    'task_net': self.task_net.state_dict(),
                    'modulation_net': self.modulation_net.state_dict(),
                    'optim_feat_net': self.optim_feat_meta.state_dict(),
                    'optim_task_net': self.optim_task_net.state_dict(),
                    'optim_modulation_net': self.optim_modulation_net.state_dict()
                }, join(self.model_path, 'GenM_intra_acc_checkpoint.tar'))
                best_intra_acc = intra_acc
            if not best_inter_acc or best_inter_acc < inter_acc:
                torch.save({
                    'feat_net': self.feat_meta.state_dict(),
                    'task_net': self.task_net.state_dict(),
                    'modulation_net': self.modulation_net.state_dict(),
                    'optim_feat_net': self.optim_feat_meta.state_dict(),
                    'optim_task_net': self.optim_task_net.state_dict(),
                    'optim_modulation_net': self.optim_modulation_net.state_dict()
                }, join(self.model_path, 'GenM_inter_acc_checkpoint.tar'))
                best_inter_acc = inter_acc

            if train_loss.item() == 0.0:
                return


    def eval(self, flags, datasets, feat_net, task_net, modulation_net):
        feat_net.eval(), task_net.eval(), modulation_net.eval()
        x, y = datasets
        x, y = x.type(torch.FloatTensor).to(flags.device), y.type(torch.LongTensor).to(flags.device)
        features = feat_net(x)
        modulation_features = modulation_net(x)
        if flags.modulation_method == 'dot_pro':
            modulated_features = dot_product()(features, modulation_features)
        elif flags.modulation_method == 'self_att':
            modulated_features, attn = ScaledDotProductAttention()(modulation_features, features, features)
            # elif flags.modulation_method == 'multihead_att':
            #     modulated_features = MultiHeadAttention()()
        else:
            print('Not defined modulation method')
            return

        logits = task_net(modulated_features)
        loss = self.loss_fn(logits, y)

        prob = F.softmax(logits, dim=1)
        pred = F.softmax(logits, dim=1).argmax(dim=1)

        return loss.item(), metric(y.detach().cpu().numpy().astype(np.int), pred.detach().cpu().numpy().astype(np.int), prob[:,1].detach().cpu().numpy())

    def eval_feature(self, flags, datasets, feat_net, task_net):
        feat_net.eval(), task_net.eval()
        x, y = datasets
        x, y = x.type(torch.FloatTensor).to(flags.device), y.type(torch.LongTensor).to(flags.device)
        features = feat_net(x)
        # modulation_features = modulation_net(x)
        # modulated_features = features * modulation_features
        logits = task_net(features)
        # logits_mod = task_net(modulated_features)
        loss = self.loss_fn(logits, y)

        prob = F.softmax(logits, dim=1)
        pred = F.softmax(logits, dim=1).argmax(dim=1)

        return loss.item(), metric(y.detach().cpu().numpy().astype(np.int), pred.detach().cpu().numpy().astype(np.int), prob[:,1].detach().cpu().numpy())

    def intra_inter_test(self, flags):
        # start writing log
        log_train = join(self.log_path, flags.GenM_test)
        log_csv(log_train,
                'auc,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'acc,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'sen,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'spec,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'feat_auc,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'feat_acc,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'feat_sen,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                'feat_spec,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                ' ,avg_intra_acc,avg_inter_acc'.format(*flags.sources, *flags.unseens, *flags.sources, *flags.unseens,
                                               *flags.sources, *flags.unseens, *flags.sources, *flags.unseens,
                                               *flags.sources, *flags.unseens, *flags.sources, *flags.unseens,
                                               *flags.sources, *flags.unseens, *flags.sources, *flags.unseens
                                               ))


        # load checkpoint
        if flags.checkpoint:
            checkpoint = torch.load(join(self.model_path, flags.checkpoint))
            self.feat_meta.load_state_dict(checkpoint['feat_net'])
            self.task_net.load_state_dict(checkpoint['task_net'])
            self.modulation_net.load_state_dict(checkpoint['modulation_net'])

        # check intra-test
        # intra_losses = [0 for _ in range(len(flags.sources))]
        intra_aucs = [0 for _ in range(len(flags.sources))]
        intra_accs = [0 for _ in range(len(flags.sources))]
        intra_sens = [0 for _ in range(len(flags.sources))]
        intra_specs = [0 for _ in range(len(flags.sources))]
        intra_feat_aucs = [0 for _ in range(len(flags.sources))]
        intra_feat_accs = [0 for _ in range(len(flags.sources))]
        intra_feat_sens = [0 for _ in range(len(flags.sources))]
        intra_feat_specs = [0 for _ in range(len(flags.sources))]
        for j in range(len(flags.sources)):
            _, [intra_aucs[j], intra_accs[j], intra_sens[j], intra_specs[j]] = self.eval(flags, self.intra_test[j],
                                                                                         self.feat_meta,
                                                                                         self.task_net,
                                                                                         self.modulation_net)
            _, [intra_feat_aucs[j], intra_feat_accs[j], intra_feat_sens[j], intra_feat_specs[j]] = self.eval_feature(
                flags, self.intra_test[j],
                self.feat_meta,
                self.task_net)
        # intra_loss = sum(intra_losses) / len(flags.sources)
        # intra_auc = sum(intra_aucs) / len(flags.sources)
        intra_acc = sum(intra_accs) / len(flags.sources)
        # intra_sen = sum(intra_sens) / len(flags.sources)
        # intra_spec = sum(intra_specs) / len(flags.sources)

        # check inter-test
        # inter_losses = [0 for _ in range(len(flags.unseens))]
        inter_aucs = [0 for _ in range(len(flags.unseens))]
        inter_accs = [0 for _ in range(len(flags.unseens))]
        inter_sens = [0 for _ in range(len(flags.unseens))]
        inter_specs = [0 for _ in range(len(flags.unseens))]
        inter_feat_aucs = [0 for _ in range(len(flags.unseens))]
        inter_feat_accs = [0 for _ in range(len(flags.unseens))]
        inter_feat_sens = [0 for _ in range(len(flags.unseens))]
        inter_feat_specs = [0 for _ in range(len(flags.unseens))]
        for j in range(len(flags.unseens)):
            _, [inter_aucs[j], inter_accs[j], inter_sens[j], inter_specs[j]] = self.eval(flags, self.inter_test[j],
                                                                                         self.feat_meta,
                                                                                         self.task_net,
                                                                                         self.modulation_net)
            _, [inter_feat_aucs[j], inter_feat_accs[j], inter_feat_sens[j], inter_feat_specs[j]] = self.eval_feature(
                flags, self.inter_test[j], self.feat_meta, self.task_net)
        # inter_loss = sum(inter_losses) / len(flags.unseens)
        # inter_auc = sum(inter_aucs) / len(flags.unseens)
        inter_acc = sum(inter_accs) / len(flags.unseens)
        # inter_sen = sum(inter_sens) / len(flags.unseens)
        # inter_spec = sum(inter_specs) / len(flags.unseens)

        # writing log and tensorboard
        log_csv(log_train,
                           '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                           '{},{},{}'.format(' ', *intra_aucs, *inter_aucs,
                                             ' ', *intra_accs, *inter_accs,
                                             ' ', *intra_sens, *inter_sens,
                                             ' ', *intra_specs, *inter_specs,
                                             ' ', *intra_feat_aucs, *inter_feat_aucs,
                                             ' ', *intra_feat_accs, *inter_feat_accs,
                                             ' ', *intra_feat_sens, *inter_feat_sens,
                                             ' ', *intra_feat_specs, *inter_feat_specs,
                                             ' ', intra_acc, inter_acc))


