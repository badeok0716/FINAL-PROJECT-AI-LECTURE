# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : relgan_instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from models.RelGAN_D import RelGAN_D
from models.RelGAN_G import RelGAN_G
from utils.helpers import get_fixed_temperature, get_losses
from utils.data_loader import GenDataIter, DisDataIter
from utils.text_process import load_dict, write_tokens, tensor_to_tokens
from utils.write_op import write_pkl


class RelGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(RelGANInstructor, self).__init__(opt)
        # generator, discriminator
        self.save_key = 'init'
        self.metric_dict = dict()

        self.gen = RelGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = RelGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size, cfg.padding_idx,
                            gpu=cfg.CUDA)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

    def _run(self):
        # ===PRE-TRAINING (GENERATOR)===
        if not cfg.gen_pretrain:
            self.save_key = 'pretrain'
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pretrain_generator: {}'.format(cfg.pretrained_gen_path))

        # # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        for adv_epoch in range(cfg.ADV_train_epoch):
            self.sig.update()
            if self.sig.adv_sig:
                self.save_key = 'adv_{}'.format(adv_epoch)
                g_loss = self.adv_train_generator(cfg.ADV_g_step)  # Generator
                d_loss = self.adv_train_discriminator(cfg.ADV_d_step)  # Discriminator
                self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature

                self.log.info(
                    'g_loss: %.4f, d_loss: %.4f, temperature: %.4f' % (g_loss, d_loss, self.gen.temperature))

                # TEST
                if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                #if adv_epoch == cfg.ADV_train_epoch - 1:
                    self.log.info('[ADV] epoch %d: g_loss: %.4f, d_loss: %.4f, %s' % (
                        adv_epoch, g_loss, d_loss, self.cal_metrics(fmt_str=True)))

                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
                break

    def _test(self):
        print('>>> Begin test...')

        self._run()
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                # ===Train===
                pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)

                # ===Test===
                #if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                if epoch == epochs - 1:
                    self.log.info('[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (
                        epoch, pre_loss, self.cal_metrics(fmt_str=True)))

                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break

    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # ===Train===
            d_out_real = self.dis(real_samples, cfg.aug_gen_real)
            d_out_fake = self.dis(gen_samples, cfg.aug_gen_fake)
            g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    def adv_train_discriminator(self, d_step):
        total_loss = 0
        for step in range(d_step):
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # ===Train===
            d_out_real = self.dis(real_samples, cfg.aug_dis_real)
            d_out_fake = self.dis(gen_samples, cfg.aug_dis_fake)
            _, d_loss = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss += d_loss.item()

        return total_loss / d_step if d_step != 0 else 0

    def update_temperature(self, i, N):
        self.gen.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()

    def cal_metrics(self, fmt_str=False):
        """
        Calculate metrics
        :param fmt_str: if return format string for logging
        """
        with torch.no_grad():
            # Prepare data for evaluation
            eval_samples = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
            gen_data = GenDataIter(eval_samples)
            gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            gen_tokens_s = tensor_to_tokens(self.gen.sample(200, 200), self.idx2word_dict)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.test_data.tokens)
            self.nll_gen.reset(self.gen, self.train_data.loader)
            self.nll_div.reset(self.gen, gen_data.loader)
            self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
            self.ppl.reset(gen_tokens)

        metric_vals = [metric.get_score() for metric in self.all_metrics]
        self.metric_dict[self.save_key] = metric_vals
        write_pkl(self.metric_dict, cfg.save_root + 'content.pkl')
        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), metric_vals[idx]) for idx, metric in enumerate(self.all_metrics)])
        else:
            return metric_vals