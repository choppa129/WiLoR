import torch
import pytorch_lightning as pl
from typing import Any, Dict, Mapping, Tuple

from yacs.config import CfgNode

from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection, weak_perspective_projection, rotmat_to_axis_angle
from ..utils.pose_utils import reconstruction_error
from ..utils.freihand_metrics import align_freihand, mesh_metrics
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import RefineNet
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from . import MANO

log = get_pylogger(__name__)

class WiLoR(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup WiLoR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            checkpoint = torch.load(
                cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS,
                map_location='cpu',
                weights_only=False,
            )
            state = checkpoint.get('state_dict', checkpoint)
            normalized = {}
            for key, value in state.items():
                key = key.removeprefix('module.').removeprefix('backbone.')
                normalized[key] = value
            if cfg.TRAIN.get('PAPER_MODE', False):
                expected = {k for k in self.backbone.state_dict()
                            if k.startswith(('patch_embed.', 'blocks.', 'last_norm.'))
                            or k == 'pos_embed'}
                missing = sorted(expected - normalized.keys())
                if missing:
                    raise ValueError(f'Incomplete ViTPose encoder checkpoint: {missing[:10]}')
            incompatible = self.backbone.load_state_dict(normalized, strict=False)
            log.info(f'ViTPose initialized; new task parameters: {incompatible.missing_keys}; '
                     f'unused checkpoint keys: {incompatible.unexpected_keys}')
        elif cfg.TRAIN.get('PAPER_MODE', False):
            raise ValueError('Paper mode requires ViTPose PRETRAINED_WEIGHTS')
            
        # Create RefineNet head
        self.refine_net = RefineNet(cfg, feat_dim=1280, upscale=3)
        
        # Create discriminator
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.mano_parameter_loss = ParameterLoss()

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None
            

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

    def on_after_backward(self):
        if not self.cfg.TRAIN.get('DEBUG_MISSING_GRADIENTS', False):
            return
        for name, param in self.named_parameters():
            if param.grad is None:
                print(param.shape)
                print(name)


    def get_parameters(self):
        all_params = list(self.refine_net.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer_name = self.cfg.TRAIN.get(
            'OPTIMIZER', 'adam' if self.cfg.TRAIN.get('PAPER_MODE', False) else 'adamw')
        optimizer_cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[optimizer_name]
        optimizer = optimizer_cls(params=param_groups,
                                        # lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL <= 0:
            return optimizer
        optimizer_disc = optimizer_cls(params=self.discriminator.parameters(),
                                            lr=self.cfg.TRAIN.LR,
                                            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        return optimizer, optimizer_disc

    def forward_step(self, batch: Dict, train: bool = False,
                     return_initial: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """
        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]
        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.backbone(x[:,:,:,32:-32]) # B, 1280, 16, 12  

    
        # Compute camera translation
        device = temp_mano_params['hand_pose'].device
        dtype = temp_mano_params['hand_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        
        
        ## Temp MANO 
        temp_mano_params['global_orient'] = temp_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        temp_mano_params['hand_pose'] = temp_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        temp_mano_params['betas'] = temp_mano_params['betas'].reshape(batch_size, -1)
        temp_mano_output  = self.mano(**{k: v for k,v in temp_mano_params.items()}, pose2rot=False)
        temp_vertices     = temp_mano_output.vertices

        # Preserve the backbone/HaMeR-style prediction so inference tools can
        # compare it directly with the WiLoR-refined output.
        initial_output = {}
        if return_initial:
            temp_keypoints_3d = temp_mano_output.joints
            initial_cam = pred_cam.clone()
            initial_cam_t = torch.stack([
                initial_cam[:, 1], initial_cam[:, 2],
                2 * focal_length[:, 0] /
                (self.cfg.MODEL.IMAGE_SIZE * initial_cam[:, 0] + 1e-9),
            ], dim=-1)
            if self.cfg.MODEL.get('PROJECTION', 'perspective') == 'weak_perspective':
                initial_keypoints_2d = weak_perspective_projection(temp_keypoints_3d, initial_cam)
            else:
                initial_keypoints_2d = perspective_projection(
                    temp_keypoints_3d, translation=initial_cam_t,
                    focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)
            initial_output = {
                'initial_pred_cam': initial_cam,
                'initial_pred_cam_t': initial_cam_t,
                'initial_pred_mano_params': {
                    k: v.clone() for k, v in temp_mano_params.items()
                },
                'initial_pred_keypoints_3d': temp_keypoints_3d.reshape(batch_size, -1, 3),
                'initial_pred_vertices': temp_vertices.reshape(batch_size, -1, 3),
                'initial_pred_keypoints_2d': initial_keypoints_2d.reshape(batch_size, -1, 2),
            }

        pred_mano_params, pred_cam = self.refine_net(vit_out, temp_vertices, pred_cam, pred_mano_feats, focal_length) 
        # Store useful regression outputs to the output dict
        
       
        output = {}
        output.update(initial_output)
        output['pred_cam'] = pred_cam
        output['pred_mano_params'] = {k: v.clone() for k,v in pred_mano_params.items()}

        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        mano_output = self.mano(**{k: v for k,v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
  
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        
        if self.cfg.MODEL.get('PROJECTION', 'perspective') == 'weak_perspective':
            pred_keypoints_2d = weak_perspective_projection(pred_keypoints_3d, pred_cam)
        else:
            pred_keypoints_2d = perspective_projection(
                pred_keypoints_3d, translation=pred_cam_t,
                focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)
        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_mano_params = output['pred_mano_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']


        batch_size = pred_mano_params['hand_pose'].shape[0]
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_mano_params = batch['mano_params']
        has_mano_params = batch['has_mano_params']
        is_axis_angle = batch['mano_params_is_axis_angle']

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)

        # Compute loss on MANO parameters
        loss_mano_params = {}
        for k, pred in pred_mano_params.items():
            gt = gt_mano_params[k].view(batch_size, -1)
            if (self.cfg.MODEL.get('PAPER_LITERAL', False)
                    and not self.cfg.TRAIN.get('OFFICIAL_LOSSES', False) and k != 'betas'):
                pred = rotmat_to_axis_angle(pred.float().reshape(batch_size, -1, 3, 3))
                if not is_axis_angle[k].all():
                    gt = rotmat_to_axis_angle(gt.float().reshape(batch_size, -1, 3, 3))
            elif is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_mano_params[k]
            loss_mano_params[k] = self.mano_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)

        loss_vertices = pred_keypoints_3d.sum() * 0.0
        if (self.cfg.TRAIN.get('PAPER_MODE', False)
                and not self.cfg.TRAIN.get('OFFICIAL_LOSSES', False)):
            loss_vertices = self.compute_vertex_loss(batch, output)

        loss = self.cfg.LOSS_WEIGHTS.get('VERTICES_3D', 0.0) * loss_vertices+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d+\
               sum([loss_mano_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_mano_params])


        # Sum coordinates per sample, then average the batch. This reduction is
        # an explicit implementation choice; the paper does not specify it.
        if (self.cfg.TRAIN.get('PAPER_MODE', False)
                and not self.cfg.TRAIN.get('OFFICIAL_LOSSES', False)):
            loss = loss / batch_size

        losses = dict(loss=loss.detach(),
                      loss_vertices=loss_vertices.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach())

        for k, v in loss_mano_params.items():
            losses['loss_' + k] = v.detach()

        if (self.cfg.TRAIN.get('PAPER_MODE', False)
                and not self.cfg.TRAIN.get('OFFICIAL_LOSSES', False)):
            for name in losses:
                if name != 'loss':
                    losses[name] = losses[name] / batch_size

        output['losses'] = losses

        return loss

    def compute_vertex_loss(self, batch: Dict, output: Dict) -> torch.Tensor:
        """Eq. (8): MANO-local vertex L1, using available complete MANO labels.

        The current tar loader supplies MANO labels, not mesh annotations. Decode
        those labels without inventing meshes from 3D keypoints. No root alignment
        is introduced here. Missing labels are selected out before MANO decoding.
        """
        pred = output['pred_vertices']
        valid = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
        for name in ('global_orient', 'hand_pose', 'betas'):
            valid &= batch['has_mano_params'][name].reshape(-1).bool()
        if not valid.any():
            return pred.sum() * 0.0
        with torch.no_grad():
            params = {}
            for name, values in batch['mano_params'].items():
                values = values[valid].float()
                if name != 'betas':
                    flags = batch['mano_params_is_axis_angle'][name][valid].bool()
                    if not (flags.all() or (~flags).all()):
                        raise ValueError('Mixed rotation representations in one tensor are unsupported')
                    if flags.all():
                        values = aa_to_rotmat(values.reshape(-1, 3))
                    values = values.reshape(int(valid.sum()), -1, 3, 3)
                params[name] = values
            target = self.mano(**params, pose2rot=False).vertices
        return (pred[valid].float() - target).abs().sum()

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        #images = 255*images.permute(0, 2, 3, 1).cpu().numpy()

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)
        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)
        if write_to_summary_writer:
            if self.logger is None:
                return
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        if self.mesh_renderer is None or self.cfg.MODEL.get('PROJECTION', 'perspective') == 'weak_perspective':
            # The legacy mesh renderer uses perspective cameras; do not show a
            # misleading overlay for a weak-perspective training projection.
            return
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        gt_keypoints_3d = batch['keypoints_3d']
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)

        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        #predictions = self.renderer(pred_keypoints_3d[:num_images],
        #                            gt_keypoints_3d[:num_images],
        #                            2 * gt_keypoints_2d[:num_images],
        #                            images=images[:num_images],
        #                            camera_translation=pred_cam_t[:num_images])
        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy())
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    hand_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            hand_pose (torch.Tensor): Regressed hand pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = hand_pose.shape[0]
        gt_hand_pose = batch['hand_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_hand_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(hand_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_mano_params = output['pred_mano_params']
        loss = self.compute_loss(batch, output, train=True)
        # Official supervised losses sum samples; DDP averages ranks. Preserve
        # the 8-rank x 128 reference scale when changing per-rank batch size.
        # Adversarial losses already average samples and are not rescaled.
        if self.cfg.TRAIN.get('OFFICIAL_LOSSES', False):
            reference_batch = self.cfg.TRAIN.get('REFERENCE_PER_RANK_BATCH_SIZE', batch_size)
            loss = loss * (reference_batch / batch_size)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_mano_params['hand_pose'].reshape(batch_size, -1), pred_mano_params['betas'].reshape(batch_size, -1))
            if (self.cfg.MODEL.get('PAPER_LITERAL', False)
                    and not self.cfg.TRAIN.get('OFFICIAL_LOSSES', False)):
                # Eq. (8) specifies an unsquared L2 norm for each example.
                loss_adv = torch.linalg.vector_norm(disc_out.float() - 1.0, dim=-1).mean()
            else:
                loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        # Error if Nan
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        output['losses']['loss'] = loss.detach()

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(mocap_batch, pred_mano_params['hand_pose'].reshape(batch_size, -1), pred_mano_params['betas'].reshape(batch_size, -1), optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if (self.cfg.GENERAL.get('ENABLE_VISUALIZATION', True)
                and self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0):
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        metrics = self.compute_validation_metrics(batch, output)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['img'].shape[0],
            sync_dist=True,
        )
        if self.cfg.GENERAL.get('ENABLE_VISUALIZATION', True):
            self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output

    @torch.no_grad()
    def compute_validation_metrics(self, batch: Dict, output: Dict) -> Dict[str, torch.Tensor]:
        """Compute crop-space 2D and root-aligned 3D reconstruction metrics."""
        pred_joints = output['pred_keypoints_3d'].float()
        gt_joints = batch['keypoints_3d'][..., :3].float()
        confidence_3d = batch['keypoints_3d'][..., 3].float()

        pred_joints = pred_joints - pred_joints[:, [0]]
        gt_joints = gt_joints - gt_joints[:, [0]]
        joint_error = torch.linalg.vector_norm(pred_joints - gt_joints, dim=-1)
        mpjpe_mm = 1000.0 * (joint_error * confidence_3d).sum() / confidence_3d.sum().clamp_min(1.0)
        # Validation runs under mixed-precision autocast, but CUDA's batched
        # SVD (used by Procrustes alignment) is not implemented for FP16.
        # Keep only this metric in FP32; the model forward pass remains mixed
        # precision.
        with torch.autocast(device_type=pred_joints.device.type, enabled=False):
            pa_mpjpe_mm = 1000.0 * reconstruction_error(pred_joints, gt_joints).mean()
            if self.cfg.get('EVAL', {}).get('FREIHAND_PROTOCOL', True):
                pa_mpjpe_mm = 1000 * torch.linalg.vector_norm(
                    align_freihand(pred_joints, gt_joints) - gt_joints, dim=-1).mean()

        pred_2d = output['pred_keypoints_2d'].float()
        gt_2d = batch['keypoints_2d'][..., :2].float()
        confidence_2d = batch['keypoints_2d'][..., 2].float()
        # Dataset convention: normalized_xy = pixel_xy / IMAGE_SIZE - 0.5.
        # One normalized unit therefore spans IMAGE_SIZE pixels, not half.
        error_2d_px = torch.linalg.vector_norm(pred_2d - gt_2d, dim=-1) * self.cfg.MODEL.IMAGE_SIZE
        reprojection_px = (error_2d_px * confidence_2d).sum() / confidence_2d.sum().clamp_min(1.0)

        metrics = {
            'val/mpjpe_mm': mpjpe_mm,
            'val/pa_mpjpe_mm': pa_mpjpe_mm,
            'val/reprojection_px': reprojection_px,
        }

        has_mano = (
            batch['has_mano_params']['global_orient'].bool()
            & batch['has_mano_params']['hand_pose'].bool()
            & batch['has_mano_params']['betas'].bool()
        )
        if has_mano.any():
            batch_size = pred_joints.shape[0]
            gt_params = {}
            for name, value in batch['mano_params'].items():
                value = value.float()
                if name != 'betas' and batch['mano_params_is_axis_angle'][name].all():
                    value = aa_to_rotmat(value.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
                gt_params[name] = value
            gt_mano = self.mano(**gt_params, pose2rot=False)
            pred_vertices = output['pred_vertices'].float() - output['pred_keypoints_3d'].float()[:, [0]]
            gt_vertices = gt_mano.vertices.float() - gt_mano.joints.float()[:, [0]]
            pve_mm = 1000.0 * torch.linalg.vector_norm(
                pred_vertices[has_mano] - gt_vertices[has_mano], dim=-1
            ).mean()
            metrics['val/pve_mm'] = pve_mm
            if self.cfg.get('EVAL', {}).get('FREIHAND_PROTOCOL', True):
                with torch.autocast(device_type=pred_vertices.device.type, enabled=False):
                    metrics.update({f'val/{k}': v for k, v in mesh_metrics(
                        pred_vertices[has_mano], gt_vertices[has_mano]).items()})

        return metrics
