# __init__.py

import os
import torch
import torch.nn as nn
from .loss import *
from .module import *
from .adain import * 
from .template import *


class Model(ModelTemplate):

    def __init__(self, model_params: ModelParams):
        super().__init__(model_params)

        # model
        self.model_illum = IlluminationModel(name='illumination-branch').to(self.device)
        self.model_specular = IlluminationModel(name='specular-branch').to(self.device)
        self.model_reflect = ReflectanceModel(mode=self.mode).to(self.device)

        # loss
        self.loss_name = ['Rec Loss', 'Reflect Loss', 'Freq Loss', 'rPPG Loss', 'Specular Loss', 'Adv Loss']
        self.init_loss()

        # Weight for Stage 1: Lambertian Initialization
        self.weight['Rec Loss'] = 3
        self.weight['Reflect Loss'] = 3
        self.weight['Freq Loss'] = 3
        self.weight['rPPG Loss'] = 0.1
        self.weight['Specular Loss'] = 0
        self.weight['Adv Loss'] = 0
        
        self.opt_illum = torch.optim.AdamW(self.model_illum.parameters(), lr=1e-4)
        self.opt_specular = torch.optim.AdamW(self.model_specular.parameters(), lr=1e-4)
        self.opt_reflect = torch.optim.AdamW(self.model_reflect.parameters(), lr=1e-4)

        # ADAIN
        mean_tensor = torch.zeros(1, 3, 1, 1, 1, device=self.device)
        std_tensor = torch.ones(1, 3, 1, 1, 1, device=self.device)
        self.adv_style_mean = nn.Parameter(mean_tensor)
        self.adv_style_std = nn.Parameter(std_tensor)
        self.optimizer_adv = torch.optim.AdamW([self.adv_style_mean, self.adv_style_std], lr=1e-4)    
        self.ood_start_epoch = 60


    def preprocess(self):
        super().preprocess()
        self.opt_illum.zero_grad()
        self.opt_reflect.zero_grad()
        self.opt_specular.zero_grad()

        # Weight for Stage 2: Specular Isolation.
        if 20 < self.epoch <= 40 :  
            self.weight['Reconstruction Loss'] = 3
            self.weight['Specular Loss'] = 5
            self.weight['rPPG Loss'] = 3
        # Weight for Stage 3: Joint Refinement
        elif self.epoch > 40:
            self.weight['Reconstruction Loss'] = 3
            self.weight['Reflectance Loss'] = 3
            self.weight['Frequency Loss'] = 3 
            self.weight['rPPG Loss'] = 3
            self.weight['Specular Loss'] = 5
        # Weight for Stage 4: Adversarial Enhancement.
        if self.epoch >= self.ood_start_epoch:
            self.weight['OOD Consistency Loss'] = 1

       
    def load_weight(self):
        self.model_illum.load(self.get_weight_path)
        self.model_specular.load(self.get_weight_path)
        self.model_reflect.load(self.get_weight_path)

    def save_weight(self):
        self.model_illum.save(self.get_weight_path)
        self.model_specular.save(self.get_weight_path)
        self.model_reflect.save(self.get_weight_path)

    def _set_grad(self, illum=True, reflect=True, specular=True):
        for p in self.model_illum.parameters(): p.requires_grad = illum
        for p in self.model_reflect.parameters(): p.requires_grad = reflect
        for p in self.model_specular.parameters(): p.requires_grad = specular

    def train(self, data):
        rgb = data['rgb_face'].to(self.device)
        augment_rgb = data['aug-rgb_face'].to(self.device)

        rgb = torch.cat([rgb, augment_rgb], dim=0)
        gt = data['gt'].to(self.device)
        gt = torch.cat([gt, gt], dim=0)

        if self.epoch <= 20:
            self._set_grad(illum=True, reflect=True, specular=False)
            v_specular = torch.zeros_like(rgb)
        elif 20 < self.epoch <= 40:
            self._set_grad(illum=False, reflect=False, specular=True)
            v_specular = self.model_specular(rgb)
        else:
            self._set_grad(illum=True, reflect=True, specular=True)
            v_specular = self.model_specular(rgb)

        v_illum = self.model_illum(rgb)
        v_reflect, signal = self.model_reflect(rgb)
        
        signal_enh = None
        v_illum_adv = None
        rgb_adv = None

        # Adversarial Training
        if self.epoch >= self.ood_start_epoch:

            self.optimizer_adv.zero_grad()

            # Freeze model params — only update adv_style_mean / adv_style_std
            for param in self.model_illum.parameters():
                param.requires_grad = False
            for param in self.model_reflect.parameters():
                param.requires_grad = False
            for param in self.model_specular.parameters():
                param.requires_grad = False

            self.model_illum.eval()
            self.model_reflect.eval()
            self.model_specular.eval()

            v_illum_adv = apply_adain_style(v_illum.detach(), self.adv_style_mean, self.adv_style_std)
            v_illum_adv = torch.clamp(v_illum_adv, 0, 1)
            rgb_adv = v_reflect.detach() * v_illum_adv 
            _, signal_adv = self.model_reflect(rgb_adv, only_rppg=True)
    
            loss_rppg_adv = rppg_loss(signal_adv, gt)

            loss_adv_total = - loss_rppg_adv 
            loss_adv_total.backward()
            self.optimizer_adv.step()

            with torch.no_grad():
                self.adv_style_mean.clamp_(-0.1, 0.1)
                self.adv_style_std.clamp_(0.9, 1.1)

            # Restore model params & training state
            for param in self.model_illum.parameters():
                param.requires_grad = True
            for param in self.model_reflect.parameters():
                param.requires_grad = True
            for param in self.model_specular.parameters():
                param.requires_grad = True

            self.model_illum.train()
            self.model_reflect.train()
            self.model_specular.train()

            v_illum_enh = apply_adain_style(v_illum.detach(), self.adv_style_mean.detach(), self.adv_style_std.detach()).detach().clamp(0, 1)
            rgb_enh = v_reflect.detach() * v_illum_enh
            _, signal_enh = self.model_reflect(rgb_enh, only_rppg=True)


        videos = (rgb, v_illum, v_reflect, v_specular)
        loss = self.loss(videos, signal, gt, signal_enh)
        
        loss.backward()
        self.opt_illum.step()
        self.opt_specular.step()
        self.opt_reflect.step()

    def inference(self, data):
        rgb = data['rgb_face'].to(self.device)
        _, signal = self.model_reflect(rgb,only_rppg=True)
        return signal


    def loss(self, videos, signal, gt, signal_ood):
        self.batch_loss['Rec Loss'] = reconstruct_loss(videos)
        self.batch_loss['Reflect Loss'] = reflect_loss(videos)
        self.batch_loss['Freq Loss'] = frequency_loss(videos)
        self.batch_loss['rPPG Loss'] = rppg_loss(signal, gt)
        self.batch_loss['Specular Loss'] = specular_loss(videos)
        if signal_ood is not None:
            self.batch_loss['Adv Loss'] = rppg_loss(signal_ood, gt)
        else:
            self.batch_loss['Adv Loss'] = torch.tensor(0.0).to(self.device)

        return self.calculate_loss()

