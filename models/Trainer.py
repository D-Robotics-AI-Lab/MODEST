import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import cv2
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from numpy.core.numeric import Inf
from models.Model import ISGNet
from utils.builder import get_losses, get_optimizer, get_schedulers, create_dir
from utils.loss import get_surface_normal
from utils.visualize import *
from utils.evaluate import compute_depth_metrics, compute_seg_metrics



class Trainer(object):

    def __init__(self, config):
        super().__init__()
        self.config = config
        ## type of predictions (full/seg/depth)
        self.type = config['Model']['type']
        ## choose the dataset (Syntodd or Clearpose)
        self.dataset_name = config["Dataset"]["dataset_name"]
        ## the number of semantic categories
        self.num_classes = len(config['Dataset'][self.dataset_name]['classes'])
        ## loss weights
        self.seg_multi = config['Trainer']['seg_multi']
        self.depth_multi = config['Trainer']['depth_multi']
        self.depth_scale_multi = config['Trainer']['depth_scale_multi']
        self.depth_grad_multi = config['Trainer']['depth_grad_multi']
        self.depth_normal_multi = config['Trainer']['depth_normal_multi']
        self.depth_error_multi = config['Trainer']['depth_error_multi']
        ## coefficient between different iterations
        self.gamma = config['Trainer']['gamma']
        ## resolution of multi-scale predictions
        self.resolutions = [48, 96, 192, 384]
        self.device = torch.device(config['Model']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        im_resize = config['Dataset'][self.dataset_name]['transforms']['im_resize']
        ## define the model
        self.model = ISGNet(
                    image_size  =   (3,im_resize,im_resize),
                    emb_dim     =   config['Model']['emb_dim'],
                    resample_dim=   config['Model']['resample_dim'],
                    read        =   config['Model']['read'],
                    nclasses    =   self.num_classes,
                    hooks       =   config['Model']['hooks'],
                    model_timm  =   config['Model']['model_timm'],
                    type        =   self.type,
                    patch_size  =   config['Model']['patch_size'],
                    pretrain    =   config['Model']['pretrain'],
                    iterations  =   config['Model']['iterations'],
                    in_chans    =   config['Dataset'][self.dataset_name]["in_chans"]
        )

        self.model.to(self.device)
        # print(self.model)

        ## loss functions
        self.loss_depth, self.loss_depth_grad, self.loss_depth_normal, self.loss_seg, self.loss_seg_iou = get_losses(config)
        self.optimizer_backbone, self.optimizer_scratch = get_optimizer(config, self.model)
        self.schedulers = get_schedulers([self.optimizer_backbone, self.optimizer_scratch])

        self.ckpt_path = config['Trainer']['ckpt_path']
        self.path_model = os.path.join('ckpt/' + self.config['Model']['path_model'], self.model.__class__.__name__)
        self.path_statis = os.path.join('ckpt/' + self.config['Model']['path_model'], 'val_statis.txt')
        create_dir(self.path_model)

    ############################ training ################################
    def train(self, train_dataloader, val_dataloader):
        epochs = self.config['Trainer']['epochs']
        ## visualization
        if self.config['wandb']['enable']:
            wandb.init(project="Your project name", entity=self.config['wandb']['username'], name=self.config['Model']['path_model'])
            wandb.config = {
                "learning_rate_backbone": self.config['Trainer']['lr_backbone'],
                "learning_rate_scratch": self.config['Trainer']['lr_scratch'],
                "epochs": epochs,
                "batch_size": self.config['Dataset'][self.dataset_name]['batch_size']
            }
        val_loss = float('inf')
        depth_eval = float('inf')

        for epoch in range(epochs):
            print("Epoch ", epoch+1)
            ## training losses
            train_loss, train_depth_all_loss, train_seg_all_loss, train_grad_loss, train_normal_loss, train_mse_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            self.model.train()
            pbar = tqdm(train_dataloader)
            pbar.set_description("Training")
            
            for index, data in enumerate(pbar):
                
                self.optimizer_backbone.zero_grad()
                self.optimizer_scratch.zero_grad()

                ## load the data
                rgb, depth_gt, seg_gt, zero_mask, loss_mask = data['rgb'].to(self.device), data['depth_gt'].to(self.device), data['seg_gt'].to(self.device), \
                    data['zero_mask'].to(self.device), data['loss_mask'].to(self.device)
                depth_min, depth_max = data['depth_min'][0].to(self.device), data['depth_max'][0].to(self.device)
                
                output_depths, output_segs = self.model(rgb)
                
                loss, depth_loss_multiscale, seg_loss_multiscale = [], [], []
                ## calculate multi-scale losses
                for iter, output_depths_iter in enumerate(output_depths):
                    iter += 1
                    for output_depth, resolution in zip(output_depths_iter, self.resolutions):
                        depth_gt_multiscale = depth_gt.clone() if resolution == 384 else \
                            F.interpolate(depth_gt.unsqueeze(1), size=(resolution, resolution), mode="bilinear", align_corners=True).squeeze(1)
                        depth_loss_multiscale.append(self.gamma * iter * self.depth_scale_multi * self.loss_depth(output_depth.squeeze(1), depth_gt_multiscale))
                        if self.loss_depth is not None:
                            depth_loss_multiscale.append(self.gamma * iter * self.depth_grad_multi * self.loss_depth_grad(output_depth, depth_gt_multiscale.unsqueeze(1)))
                        if self.loss_depth_normal is not None:
                            output_normal, _, _ = get_surface_normal(output_depth, self.dataset_name)
                            target_normal, _, _ = get_surface_normal(depth_gt_multiscale.unsqueeze(1), self.dataset_name)
                            depth_loss_multiscale.append(self.gamma * iter * self.depth_normal_multi * self.loss_depth_normal(output_normal, target_normal))

                depth_loss = sum(depth_loss_multiscale)
                loss.append(self.depth_multi * depth_loss)
                train_depth_all_loss += self.depth_multi * depth_loss.item()

                for iter, output_segs_iter in enumerate(output_segs):
                    iter += 1
                    for output_seg, resolution in zip(output_segs_iter, self.resolutions):
                        seg_gt_multiscale = seg_gt.clone() if resolution == 384 else\
                                F.interpolate(seg_gt.unsqueeze(1).float(), size=(resolution, resolution), mode="nearest").squeeze(1).long()
                        seg_loss_multiscale.append(self.gamma * iter * self.loss_seg(output_seg, seg_gt_multiscale))
                        # seg_loss_multiscale.append(self.gamma * iter * self.loss_seg_iou(output_seg, seg_gt_multiscale))
                        
                seg_loss = sum(seg_loss_multiscale)
                loss.append(self.seg_multi * seg_loss)
                train_seg_all_loss += self.seg_multi * seg_loss.item()

                losses = sum(loss)
                losses.backward()
                self.optimizer_scratch.step()
                self.optimizer_backbone.step()

                train_loss += losses.item()
                ## debug    
                if np.isnan(train_loss):
                    print('\n',
                        rgb.min().item(), rgb.max().item(),'\n',
                        depth_gt.min().item(), depth_gt.max().item(),'\n',
                        output_depths.min().item(), output_depths.max().item(),'\n',
                        loss.item(), '\n',
                        depth_loss.item(), '\n',
                        seg_loss.item(), '\n',
                    )
                    exit(0)
                ## visualization
                if self.config['wandb']['enable'] and ((index % 50 == 0 and index>0) or index==len(train_dataloader)-1):
                    wandb.log({
                        "train_loss": train_loss / (index + 1),
                        "train_depth_all_loss": train_depth_all_loss / (index + 1),
                        "train_seg_all_loss": train_seg_all_loss / (index + 1),
                        "train_grad_loss": train_grad_loss / (index + 1),
                        "train_normal_loss": train_normal_loss / (index + 1),
                        "train_mse_loss": train_mse_loss / (index + 1),
                    })
                pbar.set_postfix({'total_training_loss': train_loss/(index+1)})
            
            ## validation
            new_val_loss, new_depth_eval, new_seg_eval = self.run_eval(val_dataloader)
            ## save ckpts
            if epoch % 2 == 0:
                self.save_model(epoch)

            old_scratch_lr = [group['lr'] for group in self.optimizer_scratch.param_groups]
            old_backbone_lr = [group['lr'] for group in self.optimizer_backbone.param_groups]
            self.schedulers[0].step(new_val_loss)
            self.schedulers[1].step(new_val_loss)

            for i, group in enumerate(self.optimizer_backbone.param_groups):
                new_lr = group['lr']
                if new_lr != old_backbone_lr[i]:
                    print(f"Backbone Learning rate reduced from {old_backbone_lr[i]} to {new_lr} at epoch {epoch+1}")
            for i, group in enumerate(self.optimizer_scratch.param_groups):
                new_lr = group['lr']
                if new_lr != old_scratch_lr[i]:
                    print(f"Scratch Learning rate reduced from {old_scratch_lr[i]} to {new_lr} at epoch {epoch+1}")

        print('Finished Training')

    ############################ validation ################################
    def run_eval(self, val_dataloader):
        """
            Evaluate the model on the validation set and visualize some results
            on wandb
            :- val_dataloader -: torch dataloader
        """
        val_size = len(val_dataloader)
        val_loss = 0.
        val_depth_loss, val_seg_loss, val_mse_loss = 0.0, 0.0, 0.0
        ## evaluation metrics
        MAE_all, RMSE_all, REL_all, DELTA105_all, DELTA110_all, DELTA125_all = [], [], [], [], [], []
        MAE_mask_all, RMSE_mask_all, REL_mask_all, DELTA105_mask_all, DELTA110_mask_all, DELTA125_mask_all = [], [], [], [], [], []
        IoU_all, mAP_all = [], []
        
        self.model.eval()
        depth_eval, seg_eval = None, None
        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            pbar.set_description("Validation")
            for index, data in enumerate(pbar):

                rgb, depth_gt, seg_gt, zero_mask, loss_mask = data['rgb'].to(self.device), data['depth_gt'].to(self.device), data['seg_gt'].to(self.device), \
                    data['zero_mask'].to(self.device), data['loss_mask'].to(self.device)
                depth_min, depth_max = data['depth_min'][0].to(self.device), data['depth_max'][0].to(self.device)
                output_depths, output_segs = self.model(rgb)

                loss = []
                depth_loss_multiscale, seg_loss_multiscale = [], []

                for iter, output_depths_iter in enumerate(output_depths):
                    iter += 1
                    for output_depth, resolution in zip(output_depths_iter, self.resolutions):
                        depth_gt_multiscale = depth_gt.clone() if resolution == 384 else \
                            F.interpolate(depth_gt.unsqueeze(1), size=(resolution, resolution), mode="bilinear", align_corners=True).squeeze(1)
                        depth_loss_multiscale.append(self.gamma * iter * self.loss_depth(output_depth.squeeze(1), depth_gt_multiscale))
                        if self.loss_depth is not None:
                            depth_loss_multiscale.append(self.gamma * iter * self.depth_grad_multi * self.loss_depth_grad(output_depth, depth_gt_multiscale.unsqueeze(1)))
                        if self.loss_depth_normal is not None:
                            output_normal, _, _ = get_surface_normal(output_depth, self.dataset_name)
                            target_normal, _, _ = get_surface_normal(depth_gt_multiscale.unsqueeze(1), self.dataset_name)
                            depth_loss_multiscale.append(self.gamma * iter * self.depth_normal_multi * self.loss_depth_normal(output_normal, target_normal))

                depth_loss = sum(depth_loss_multiscale)
                loss.append(self.depth_multi * depth_loss)
                val_depth_loss += self.depth_multi * depth_loss.item()

                MAE, RMSE, REL, DELTA105, DELTA110, DELTA125, MAE_, RMSE_, REL_, DELTA105_, DELTA110_, DELTA125_ \
                    = compute_depth_metrics(output_depths[-1][-1].squeeze(1), depth_gt, depth_min, depth_max, zero_masks=zero_mask, \
                                            denorm=True, gt_masks=seg_gt, num_classes=self.num_classes)
                MAE_all.append(MAE)
                RMSE_all.append(RMSE)
                REL_all.append(REL)
                DELTA105_all.append(DELTA105)
                DELTA110_all.append(DELTA110)
                DELTA125_all.append(DELTA125)
                MAE_mask_all.append(MAE_)
                RMSE_mask_all.append(RMSE_)
                REL_mask_all.append(REL_)
                DELTA105_mask_all.append(DELTA105_)
                DELTA110_mask_all.append(DELTA110_)
                DELTA125_mask_all.append(DELTA125_)
                
                for iter, output_segs_iter in enumerate(output_segs):
                    iter += 1
                    for output_seg, resolution in zip(output_segs_iter, self.resolutions):
                        seg_gt_multiscale = seg_gt.clone() if resolution == 384 else \
                            F.interpolate(seg_gt.unsqueeze(1).float(), size=(resolution, resolution), mode="nearest").squeeze(1).long()
                        seg_loss_multiscale.append(self.gamma * iter * self.loss_seg(output_seg, seg_gt_multiscale))
               
                seg_loss = sum(seg_loss_multiscale)
                loss.append(self.seg_multi * seg_loss)
                val_seg_loss += self.seg_multi * seg_loss
                
                IoU, mAP = compute_seg_metrics(output_segs[-1][-1], seg_gt, num_classes=self.num_classes)
                IoU_all.append(IoU)
                mAP_all.append(mAP)

                losses = sum(loss)
                val_loss += losses.item()
                pbar.set_postfix({'total_validation_loss': val_loss/(index+1)})
                
                ## visualization
                if index==0:
                    rgb_visual = rgb
                    depth_gt_visual = depth_gt
                    depth_pred_visual = output_depths
                    seg_gt_visual = seg_gt
                    seg_pred_visual = output_segs
                    zero_mask_visual = zero_mask

            ## visualization
            if self.config['wandb']['enable']:
                wandb.log({
                        "val_loss": val_loss / val_size,
                        "val_depth_all_loss": val_depth_loss / val_size,
                        "val_mse_loss": val_mse_loss / val_size,
                        "val_seg_all_loss": val_seg_loss / val_size
                    })
                
                self.img_logger(rgb_visual, depth_gt_visual, seg_gt_visual, depth_pred_visual, seg_pred_visual, zero_mask_visual)

            if len(MAE_all) != 0:
                MAE_mean = sum(MAE_all) / len(MAE_all)
                RMSE_mean = sum(RMSE_all) / len(RMSE_all)
                REL_mean = sum(REL_all) / len(REL_all)
                DELTA105_mean = sum(DELTA105_all) / len(DELTA105_all)
                DELTA110_mean = sum(DELTA110_all) / len(DELTA110_all)
                DELTA125_mean = sum(DELTA125_all) / len(DELTA125_all)
                MAE_mask_mean = sum(MAE_mask_all) / len(MAE_mask_all)
                RMSE_mask_mean = sum(RMSE_mask_all) / len(RMSE_mask_all)
                REL_mask_mean = sum(REL_mask_all) / len(REL_mask_all)
                DELTA105_mask_mean = sum(DELTA105_mask_all) / len(DELTA105_mask_all)
                DELTA110_mask_mean = sum(DELTA110_mask_all) / len(DELTA110_mask_all)
                DELTA125_mask_mean = sum(DELTA125_mask_all) / len(DELTA125_mask_all)
                depth_eval = MAE_mean + RMSE_mean + REL_mean if self.dataset_name == "syntodd" else MAE_mask_mean + RMSE_mask_mean + REL_mask_mean
                print("val_RMSE: ", RMSE_mean, '\t', "val_MAE: ", MAE_mean, '\t', "val_REL: ", REL_mean, '\t', "val_105: ", DELTA105_mean,\
                      '\t', "val_110: ", DELTA110_mean, '\t', "val_125: ", DELTA125_mean)
            
            if len(IoU_all) != 0:
                IOU_mean = sum(IoU_all)/len(IoU_all)
                mAP_mean = sum(mAP_all)/len(mAP_all)
                seg_eval = IOU_mean+mAP_mean
                print("val_mAP: ", mAP_mean, '\t', "val_IoU: ", IOU_mean)
                with open(self.path_statis, 'a') as f:
                    f.write(f"mAP: {mAP_mean:.5f} \t IoU: {IOU_mean:.5f} \n\n")

        return val_loss / val_size, depth_eval, seg_eval


    ############################ test ################################
    def test(self, test_dataloader):
        ## load model weights
        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("load ckpt successfully")
        self.model.eval()

        with torch.no_grad():
            MAE_all, RMSE_all, REL_all, DELTA105_all, DELTA110_all, DELTA125_all = [], [], [], [], [], []
            MAE_mask_all, RMSE_mask_all, REL_mask_all, DELTA105_mask_all, DELTA110_mask_all, DELTA125_mask_all = [], [], [], [], [], []
            IoU_all, mAP_all = [], []
            pbar = tqdm(test_dataloader)
            pbar.set_description("Test")
            for index, data in enumerate(pbar):
                
                rgb, depth_gt, seg_gt, zero_mask, loss_mask = data['rgb'].to(self.device), data['depth_gt'].to(self.device), data['seg_gt'].to(self.device), \
                            data['zero_mask'].to(self.device), data['loss_mask'].to(self.device)
                depth_min, depth_max = data['depth_min'][0].to(self.device), data['depth_max'][0].to(self.device)
                output_depths, output_segs = self.model(rgb)

                MAE, RMSE, REL, DELTA105, DELTA110, DELTA125, MAE_, RMSE_, REL_, DELTA105_, DELTA110_, \
                            DELTA125_ = compute_depth_metrics(output_depths[-1][-1].squeeze(1), depth_gt, depth_min, depth_max, \
                                                            zero_masks=zero_mask, denorm=True, gt_masks=seg_gt, num_classes=3)
                MAE_all.append(MAE)
                RMSE_all.append(RMSE)
                REL_all.append(REL)
                DELTA105_all.append(DELTA105)
                DELTA110_all.append(DELTA110)
                DELTA125_all.append(DELTA125)
                MAE_mask_all.append(MAE_)
                RMSE_mask_all.append(RMSE_)
                REL_mask_all.append(REL_)
                DELTA105_mask_all.append(DELTA105_)
                DELTA110_mask_all.append(DELTA110_)
                DELTA125_mask_all.append(DELTA125_)

                IoU, mAP = compute_seg_metrics(output_segs[-1][-1], seg_gt)
                IoU_all.append(IoU)
                mAP_all.append(mAP)

            # img_logger(rgb, depth_gt, seg_gt, output_depths, output_segs, zero_mask)
                
            if len(MAE_all) != 0:
                print("test_MAE: ", sum(MAE_all)/len(MAE_all), '\t', "test_RMSE: ", sum(RMSE_all)/len(RMSE_all), '\t', "test_REL: ", sum(REL_all)/len(REL_all))
                print("mask_MAE: ", sum(MAE_mask_all)/len(MAE_mask_all), '\t', "mask_RMSE: ", sum(RMSE_mask_all)/len(RMSE_mask_all), '\t', "mask_REL: ", sum(REL_mask_all)/len(REL_mask_all))
            if len(IoU_all) != 0:
                print("test_IoU: ", sum(IoU_all)/len(IoU_all), '\t', "test_mAP: ", sum(mAP_all)/len(mAP_all))


    ############################ inference ################################
    def inference(self, image_path):
        ## load model weights
        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("load ckpt successfully")
        self.model.eval()

        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])
        image = transform(image)
        image = image.unsqueeze(0)


        with torch.no_grad():
            image = image.to(self.device)
            output_depth, output_seg = self.model(image)
            depth_visual = vis_depth(output_depth[-1][-1].cpu().squeeze(0).squeeze(0))
            seg_visual = vis_seg(output_seg[-1][-1].cpu().squeeze(0), image.cpu().squeeze(0).permute(1, 2, 0), 'syntodd')
            depth_image = Image.fromarray(depth_visual)
            seg_image = Image.fromarray(seg_visual)
            depth_image.save('results/depth.png')
            seg_image.save('results/seg.png')


    ############################ save ckpts ################################
    def save_model(self, name=None):
        if name == None:
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                        'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
                        }, self.path_model+'.p')
            print('Model saved at : {}'.format(self.path_model))
        else:
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                        'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
                        }, self.path_model + str(name) + '.p')
            print('Model saved at : {}'.format(self.path_model))

    ############################ visualization ################################
    def img_logger(self, rgb, depth_gt, seg_gt, depth_pred, seg_pred, zero_mask):
        nb_to_show = self.config['wandb']['images_to_show'] if self.config['wandb']['images_to_show'] <= len(rgb) else len(rgb)
        seg_truths_all, depth_truths_all = [], []
        seg_preds_all, depth_preds_all, error_all = [], [], [], [], []
        tmp = rgb[:nb_to_show].detach()
        imgs = [vis_rgb(img, self.dataset_name) for img in tmp]

        depth_truths = depth_gt[:nb_to_show]
        for depth_truth in depth_truths:
            depth_visual_gt = vis_depth(depth_truth)
            depth_truths_all.append(depth_visual_gt)

        for output_depths_iter in depth_pred:
            depth_preds_iter, error_iter = [], []
            output_depth_iter = output_depths_iter[-1][:nb_to_show].squeeze(1)
            for pred, truth in zip(output_depth_iter, depth_gt):
                error_visual = vis_error_map(pred, truth, 'depth')
                error_iter.append(error_visual)
                depth_visual_pred = vis_depth(pred)
                depth_preds_iter.append(depth_visual_pred)
            depth_preds_all.append(depth_preds_iter)
            error_all.append(error_iter)
                    
        seg_truths = seg_gt[:nb_to_show]
        for truths, img in zip(seg_truths, imgs):
            seg_visual_gt = vis_seg_gt(truths, img, self.dataset_name)
            seg_truths_all.append(seg_visual_gt)
        
        for output_segs_iter in seg_pred:
            seg_preds_iter = []
            preds = output_segs_iter[-1][:nb_to_show]
            for pred, img in zip(preds, imgs):
                seg_visual_pred = vis_seg(pred, img, self.dataset_name)
                seg_preds_iter.append(seg_visual_pred)
            seg_preds_all.append(seg_preds_iter)

        seg_truths_all = np.array(seg_truths_all)
        seg_preds_all = np.array(seg_preds_all)

        output_dim = (int(self.config['wandb']['im_w']), int(self.config['wandb']['im_h']))

        wandb.log({
            "img": [wandb.Image(Image.fromarray(im).resize(output_dim), caption='img_{}'.format(i+1)) for i, im in enumerate(imgs)]
        })

        wandb.log({
            "depth_truths": [wandb.Image(cv2.resize(im, output_dim), caption=f'depth_truths_{i+1}') for i, im in enumerate(depth_truths_all)],
            "depth_preds": [wandb.Image(Image.fromarray(depth_preds_all[j][i]).resize(output_dim), caption=f'depth_preds_{i+1}_{j+1}') \
                            for i in range(len(depth_preds_all[0])) for j in range(len(depth_preds_all))]
        })

        wandb.log({
            "seg_truths": [wandb.Image(Image.fromarray(im).resize(output_dim), caption=f'seg_truths_{i+1}') for i, im in enumerate(seg_truths_all)],
            "seg_preds": [wandb.Image(Image.fromarray(seg_preds_all[j][i]).resize(output_dim), caption=f'seg_preds_{i+1}_{j+1}') \
                            for i in range(len(seg_preds_all[0])) for j in range(len(seg_preds_all))]
        })


    def isnan(self, x):
        return x != x
